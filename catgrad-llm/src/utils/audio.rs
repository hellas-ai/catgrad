use crate::model_media::{AUDIO_FEATURE_SIZE, PreparedAudioFeatures};
use crate::{LLMError, Result};
use hound::{SampleFormat, WavReader};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::path::Path;

// This has Gemma4 specific audio constants as it is only used by the Gemma 4 multimodal models
// Clean up when other models support audio
pub const AUDIO_SAMPLE_RATE: u32 = 16_000;
pub const AUDIO_FRAME_LENGTH: usize = 320;
pub const AUDIO_HOP_LENGTH: usize = 160;
pub const AUDIO_FFT_LENGTH: usize = 512;
pub const AUDIO_MEL_FLOOR: f32 = 1e-3;
const GEMMA4_AUDIO_MAX_SAMPLES: usize = 480_000;
const GEMMA4_AUDIO_PAD_TO_MULTIPLE_OF: usize = 128;

pub fn load_wav_file(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        WavReader::open(path).map_err(|err| LLMError::IoError(std::io::Error::other(err)))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(LLMError::UnsupportedWireConversion(format!(
            "expected mono wav input, found {} channels",
            spec.channels
        )));
    }
    if spec.sample_rate != AUDIO_SAMPLE_RATE {
        return Err(LLMError::UnsupportedWireConversion(format!(
            "expected {AUDIO_SAMPLE_RATE}Hz wav input, found {}Hz",
            spec.sample_rate
        )));
    }

    let samples = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|err| LLMError::IoError(std::io::Error::other(err)))?,
        SampleFormat::Int => {
            let scale = ((1i64 << (spec.bits_per_sample.saturating_sub(1) as u32)) - 1) as f32;
            if scale <= 0.0 {
                return Err(LLMError::UnsupportedWireConversion(format!(
                    "unsupported wav bit depth {}",
                    spec.bits_per_sample
                )));
            }
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|sample| sample as f32 / scale)
                        .map_err(|err| LLMError::IoError(std::io::Error::other(err)))
                })
                .collect::<Result<Vec<_>>>()?
        }
    };

    Ok(samples)
}

pub fn gemma4_audio_mel_frame_count(num_samples: usize) -> usize {
    let padded_samples = num_samples + AUDIO_FRAME_LENGTH / 2;
    let frame_size_for_unfold = AUDIO_FRAME_LENGTH + 1;
    if padded_samples < frame_size_for_unfold {
        0
    } else {
        (padded_samples - frame_size_for_unfold) / AUDIO_HOP_LENGTH + 1
    }
}

pub fn compute_log_mel_spectrogram(waveform: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let waveform = if waveform.len() > GEMMA4_AUDIO_MAX_SAMPLES {
        &waveform[..GEMMA4_AUDIO_MAX_SAMPLES]
    } else {
        waveform
    };
    let padded_len = round_up_to_multiple(waveform.len(), GEMMA4_AUDIO_PAD_TO_MULTIPLE_OF);
    let mut padded_input = vec![0.0f32; padded_len];
    padded_input[..waveform.len()].copy_from_slice(waveform);
    let mut sample_mask = vec![false; padded_len];
    sample_mask[..waveform.len()].fill(true);

    let frame_size_for_unfold = AUDIO_FRAME_LENGTH + 1;
    let pad_left = AUDIO_FRAME_LENGTH / 2;
    let mut padded_waveform = vec![0.0f32; pad_left + padded_len];
    padded_waveform[pad_left..].copy_from_slice(&padded_input);
    let mut padded_mask = vec![false; pad_left + padded_len];
    padded_mask[pad_left..].copy_from_slice(&sample_mask);
    if padded_waveform.len() < frame_size_for_unfold {
        return Ok((Vec::new(), Vec::new(), 0));
    }

    let num_frames = gemma4_audio_mel_frame_count(padded_len);
    let window = hann_window();
    let mel_filters = create_mel_filterbank(
        AUDIO_FEATURE_SIZE,
        AUDIO_FFT_LENGTH,
        AUDIO_SAMPLE_RATE as f32,
        0.0,
        AUDIO_SAMPLE_RATE as f32 / 2.0,
    );
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(AUDIO_FFT_LENGTH);

    let mut features = Vec::with_capacity(num_frames * AUDIO_FEATURE_SIZE);
    let mut mask = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let start = frame_idx * AUDIO_HOP_LENGTH;
        let raw_frame = &padded_waveform[start..start + frame_size_for_unfold];
        let mut windowed = raw_frame[..AUDIO_FRAME_LENGTH]
            .iter()
            .zip(window.iter())
            .map(|(sample, coeff)| Complex32::new(sample * coeff, 0.0))
            .collect::<Vec<_>>();
        windowed.resize(AUDIO_FFT_LENGTH, Complex32::new(0.0, 0.0));
        fft.process(&mut windowed);

        let magnitude = windowed[..AUDIO_FFT_LENGTH / 2 + 1]
            .iter()
            .map(|complex| complex.norm())
            .collect::<Vec<_>>();

        let valid = padded_mask[start + frame_size_for_unfold - 1];
        for filter in &mel_filters {
            let mel = filter
                .iter()
                .enumerate()
                .fold(0.0f32, |acc, (freq_idx, coeff)| {
                    acc + magnitude[freq_idx] * coeff
                });
            features.push(if valid {
                (mel + AUDIO_MEL_FLOOR).ln()
            } else {
                0.0
            });
        }
        mask.push(if valid { 0.0 } else { 1.0 });
    }

    Ok((features, mask, num_frames))
}

pub fn prepare_audio_features(path: &Path) -> Result<PreparedAudioFeatures> {
    let waveform = load_wav_file(path)?;
    let (features, mask, num_mel_frames) = compute_log_mel_spectrogram(&waveform)?;
    if num_mel_frames == 0 {
        return Err(LLMError::UnsupportedWireConversion(
            "audio input produced no log-mel frames".to_string(),
        ));
    }
    let valid_mel_frames = mask.iter().filter(|&&value| value == 0.0).count();
    Ok(PreparedAudioFeatures {
        feature_shape: vec![1, num_mel_frames, AUDIO_FEATURE_SIZE],
        features,
        mask_shape: vec![1, num_mel_frames],
        mask,
        num_mel_frames,
        valid_mel_frames,
    })
}

pub fn prepare_gemma4_audio_input(
    audio_path: &Path,
    config_json: &serde_json::Value,
) -> Result<crate::models::gemma4::Gemma4PreparedAudioInput> {
    let prepared = prepare_audio_features(audio_path)?;
    Ok(crate::models::gemma4::prepare_gemma4_audio_input_from_features(prepared, config_json)?)
}

fn hann_window() -> Vec<f32> {
    let arg = std::f32::consts::PI * 2.0 / AUDIO_FRAME_LENGTH as f32;
    (0..AUDIO_FRAME_LENGTH)
        .map(|idx| 0.5 - 0.5 * (arg * idx as f32).cos())
        .collect()
}

fn create_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    min_frequency: f32,
    max_frequency: f32,
) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let all_freqs = (0..n_freqs)
        .map(|idx| idx as f32 * sample_rate / n_fft as f32)
        .collect::<Vec<_>>();

    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let min_mel = hz_to_mel(min_frequency);
    let max_mel = hz_to_mel(max_frequency);
    let mut freq_points = Vec::with_capacity(n_mels + 2);
    for idx in 0..(n_mels + 2) {
        let mel = min_mel + (max_mel - min_mel) * idx as f32 / (n_mels + 1) as f32;
        freq_points.push(mel_to_hz(mel));
    }

    let mut filterbank = vec![vec![0.0; n_freqs]; n_mels];
    for mel_idx in 0..n_mels {
        let left = freq_points[mel_idx];
        let center = freq_points[mel_idx + 1];
        let right = freq_points[mel_idx + 2];
        for (freq_idx, &freq) in all_freqs.iter().enumerate() {
            if freq >= left && freq <= center && center > left {
                filterbank[mel_idx][freq_idx] = (freq - left) / (center - left);
            } else if freq > center && freq <= right && right > center {
                filterbank[mel_idx][freq_idx] = (right - freq) / (right - center);
            }
        }
    }
    filterbank
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    if value == 0 || multiple == 0 {
        value
    } else {
        value.div_ceil(multiple) * multiple
    }
}
