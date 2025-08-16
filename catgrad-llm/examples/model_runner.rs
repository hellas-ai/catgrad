use catgrad_llm::run::*;
use catgrad_llm::serve::*;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The Loader trait splits models into the core "tensor" loop (Runner) and decoding/chatML
    // formatting (Tokenizer).
    let loader = ModelLoader::new("Qwen/Qwen3-0.6B", true).unwrap();
    let mut runner = loader.load_runner()?;
    let tokenizer = loader.load_tokenizer()?;

    // Make some message context
    let system_message = Message {
        role: "system".to_string(),
        content: "You are a helpful chat assistant".to_string(),
    };

    let prompt_message = Message {
        role: "user".to_string(),
        content: "What is 2+2?".to_string(),
    };
    let messages = vec![system_message, prompt_message];

    // Use runner to generate new tokens after given context.
    // This handles ChatML transparently via tokenizer.encode_messages
    let context = tokenizer.encode_messages(messages)?;
    for token in runner.complete(context) {
        print!("{}", tokenizer.decode(vec![token])?);
        let _ = std::io::stdout().flush();
    }

    Ok(())
}
