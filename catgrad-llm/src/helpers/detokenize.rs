//! Incremental detokenization for streamed model output.
//!
//! # Example
//!
//! ```rust
//! use catgrad_llm::{Detokenizer, Result};
//! use tokenizers::models::wordlevel::WordLevel;
//! use tokenizers::Tokenizer;
//!
//! # fn main() -> Result<()> {
//! let model = WordLevel::builder()
//!     .vocab(
//!         [
//!             ("[UNK]".to_string(), 0u32),
//!             ("hello".to_string(), 1u32),
//!             ("world".to_string(), 2u32),
//!         ]
//!         .into_iter()
//!         .collect(),
//!     )
//!     .unk_token("[UNK]".to_string())
//!     .build()
//!     .unwrap();
//! let tokenizer = Tokenizer::new(model);
//!
//! let mut detokenizer = Detokenizer::from_tokenizer(&tokenizer, &[99]);
//!
//! let _ = detokenizer.push_tokens(&[1])?;
//! let _ = detokenizer.push_tokens(&[2, 99])?;
//! assert!(detokenizer.is_stopped());
//! # Ok(())
//! # }
//! ```

use crate::{LLMError, Result};
use tokenizers::tokenizer::Tokenizer;

type TokenDecoder<'a> = dyn for<'b> Fn(&'b [i32]) -> Result<String> + Send + Sync + 'a;

/// Decodes token ids with a Hugging Face tokenizer.
pub fn detokenize_tokens(tokenizer: &Tokenizer, token_ids: &[i32]) -> Result<String> {
    let token_ids: Vec<u32> = token_ids
        .iter()
        .map(|&token| {
            u32::try_from(token).map_err(|_| {
                LLMError::TokenizerError(format!("negative token id {token} cannot be decoded"))
            })
        })
        .collect::<Result<_>>()?;
    Ok(tokenizer.decode(&token_ids, false)?)
}

/// Incrementally detokenizes token batches into text deltas.
pub struct Detokenizer<'a> {
    decode_tokens: Box<TokenDecoder<'a>>,
    stop_token_ids: Vec<i32>,
    tokens: Vec<i32>,
    decoded: String,
    stopped: bool,
}

impl<'a> Detokenizer<'a> {
    /// Creates a detokenizer from a token-decoding function and stop ids.
    pub fn new<F>(decode_tokens: F, stop_token_ids: &[i32]) -> Self
    where
        F: for<'b> Fn(&'b [i32]) -> Result<String> + Send + Sync + 'a,
    {
        Self {
            decode_tokens: Box::new(decode_tokens),
            stop_token_ids: stop_token_ids.to_vec(),
            tokens: Vec::new(),
            decoded: String::new(),
            stopped: false,
        }
    }

    /// Creates a detokenizer backed by a Hugging Face tokenizer.
    pub fn from_tokenizer(tokenizer: &'a Tokenizer, stop_token_ids: &[i32]) -> Self {
        Self::new(
            move |token_ids| detokenize_tokens(tokenizer, token_ids),
            stop_token_ids,
        )
    }

    /// Pushes more token ids and returns only the newly decoded text.
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

        let next_decoded = (self.decode_tokens)(&self.tokens)?;
        let delta = next_decoded
            .strip_prefix(self.decoded.as_str())
            .unwrap_or(next_decoded.as_str())
            .to_string();
        self.decoded = next_decoded;
        Ok(delta)
    }

    /// Returns the full decoded text accumulated so far.
    pub fn finish(self) -> String {
        self.decoded
    }

    /// Returns whether a stop token has been seen.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

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
        let mut decoder = Detokenizer::new(fake_decoder, &[]);
        assert_eq!(decoder.push_tokens(&[1]).unwrap(), "alpha");
        assert_eq!(decoder.push_tokens(&[2]).unwrap(), "beta");
        assert_eq!(decoder.finish(), "alphabeta");
    }

    #[test]
    fn suppresses_stop_tokens() {
        let mut decoder = Detokenizer::new(fake_decoder, &[99]);
        assert_eq!(decoder.push_tokens(&[1, 99, 2]).unwrap(), "alpha");
        assert!(decoder.is_stopped());
        assert_eq!(decoder.push_tokens(&[3]).unwrap(), "");
        assert_eq!(decoder.finish(), "alpha");
    }

    #[test]
    fn handles_multi_token_batches() {
        let mut decoder = Detokenizer::new(fake_decoder, &[]);
        assert_eq!(decoder.push_tokens(&[1, 2]).unwrap(), "alphabeta");
        assert_eq!(decoder.push_tokens(&[3]).unwrap(), "gamma");
    }

    #[test]
    fn from_tokenizer_uses_tokenizer_decode() {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0u32),
                    ("hello".to_string(), 1u32),
                    ("world".to_string(), 2u32),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let tokenizer = Tokenizer::new(model);

        let mut detokenizer = Detokenizer::from_tokenizer(&tokenizer, &[99]);
        let _ = detokenizer.push_tokens(&[1]).unwrap();
        let _ = detokenizer.push_tokens(&[2, 99]).unwrap();

        assert!(detokenizer.is_stopped());
    }
}
