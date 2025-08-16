//! Abstract interfaces for serving LLMs
use crate::Result;

/// Message type for use with instruct models
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A language model has a settable internal context from which it can generate new tokens.
pub trait LM<T>: Iterator<Item = T> {
    fn set_context(&mut self, context: Vec<T>);

    fn complete(&mut self, context: Vec<T>) -> impl Iterator<Item = T> {
        self.set_context(context);
        self
    }
}

/// A *loader* is conceptually a pair of language model and supporting code (tokenizers, ChatML
/// templates, etc.)
pub trait Loader<Token, L: LM<Token>, T: Tokenizer<Token>> {
    fn load_runner(&self) -> Result<L>;
    fn load_tokenizer(&self) -> Result<T>;
}

/// A [`Tokenizer`] translates between tokens and strings
pub trait Tokenizer<Token> {
    fn encode(&self, content: String) -> Result<Vec<Token>>;
    fn decode(&self, tokens: Vec<Token>) -> Result<String>;
}

/// A [`Tokenizer`] which is aware of message structure
pub trait ChatTokenizer<Token> {
    fn encode_messages(&self, messages: Vec<Message>) -> Result<Vec<Token>>;
}
