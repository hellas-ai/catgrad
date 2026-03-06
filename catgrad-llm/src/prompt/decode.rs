use crate::Result;
use std::sync::Arc;

pub trait TokenDecoder {
    fn decode_tokens(&self, tokens: &[i32]) -> Result<String>;
}

impl TokenDecoder for crate::run::ModelTokenizer {
    fn decode_tokens(&self, tokens: &[i32]) -> Result<String> {
        use crate::types::Tokenizer;
        self.decode(tokens.to_vec())
    }
}

impl TokenDecoder for fn(&[i32]) -> Result<String> {
    fn decode_tokens(&self, tokens: &[i32]) -> Result<String> {
        (self)(tokens)
    }
}

impl TokenDecoder for &crate::run::ModelTokenizer {
    fn decode_tokens(&self, tokens: &[i32]) -> Result<String> {
        use crate::types::Tokenizer;
        self.decode(tokens.to_vec())
    }
}

impl<T> TokenDecoder for Arc<T>
where
    T: TokenDecoder + ?Sized,
{
    fn decode_tokens(&self, tokens: &[i32]) -> Result<String> {
        self.as_ref().decode_tokens(tokens)
    }
}

pub struct StreamingDecoder<D> {
    decoder: D,
    stop_token_ids: Vec<i32>,
    tokens: Vec<i32>,
    decoded: String,
    stopped: bool,
}

impl<D> StreamingDecoder<D>
where
    D: TokenDecoder,
{
    pub fn new(decoder: D, stop_token_ids: &[i32]) -> Self {
        Self {
            decoder,
            stop_token_ids: stop_token_ids.to_vec(),
            tokens: Vec::new(),
            decoded: String::new(),
            stopped: false,
        }
    }

    pub fn push_tokens(&mut self, tokens: &[i32]) -> Result<String> {
        if self.stopped {
            return Ok(String::new());
        }

        let start_len = self.tokens.len();
        for &token in tokens {
            if self.stop_token_ids.contains(&token) {
                self.stopped = true;
                break;
            }
            self.tokens.push(token);
        }

        if self.tokens.len() == start_len {
            return Ok(String::new());
        }

        let next_decoded = self.decoder.decode_tokens(&self.tokens)?;
        let delta = next_decoded
            .strip_prefix(self.decoded.as_str())
            .unwrap_or(next_decoded.as_str())
            .to_string();
        self.decoded = next_decoded;
        Ok(delta)
    }

    pub fn finish(self) -> String {
        self.decoded
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_decoder(tokens: &[i32]) -> Result<String> {
        Ok(tokens
            .iter()
            .map(|token| match token {
                1 => "alpha",
                2 => "beta",
                3 => "gamma",
                _ => "?",
            })
            .collect())
    }

    #[test]
    fn emits_only_incremental_delta() {
        let mut decoder = StreamingDecoder::new(fake_decoder as fn(&[i32]) -> Result<String>, &[]);
        assert_eq!(decoder.push_tokens(&[1]).unwrap(), "alpha");
        assert_eq!(decoder.push_tokens(&[2]).unwrap(), "beta");
        assert_eq!(decoder.finish(), "alphabeta");
    }

    #[test]
    fn suppresses_stop_tokens() {
        let mut decoder =
            StreamingDecoder::new(fake_decoder as fn(&[i32]) -> Result<String>, &[99]);
        assert_eq!(decoder.push_tokens(&[1, 99, 2]).unwrap(), "alpha");
        assert!(decoder.is_stopped());
        assert_eq!(decoder.push_tokens(&[3]).unwrap(), "");
        assert_eq!(decoder.finish(), "alpha");
    }

    #[test]
    fn handles_multi_token_batches() {
        let mut decoder = StreamingDecoder::new(fake_decoder as fn(&[i32]) -> Result<String>, &[]);
        assert_eq!(decoder.push_tokens(&[1, 2]).unwrap(), "alphabeta");
        assert_eq!(decoder.push_tokens(&[3]).unwrap(), "gamma");
    }
}
