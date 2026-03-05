use catgrad_llm::run::*;
use catgrad_llm::types::*;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The Loader trait splits models into the core "tensor" loop (Runner) and decoding/chatML
    // formatting (Tokenizer).
    let loader = ModelLoader::new("Qwen/Qwen3-0.6B", true).unwrap();
    let mut runner = loader.load_runner()?;
    let tokenizer = loader.load_tokenizer()?;

    // Make some message context
    let system_message = Message::text("system", "You are a helpful chat assistant");

    let user_message = Message::text("user", "What is 2+2?");
    let messages = vec![system_message.clone(), user_message];

    // Use runner to generate new tokens after given context.
    // This handles ChatML transparently via tokenizer.encode_messages
    let context = tokenizer.encode_messages(messages, vec![])?;
    for token in runner.complete(context) {
        print!("{}", tokenizer.decode(vec![token])?);
        let _ = std::io::stdout().flush();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Change user_message to modify context history

    println!("\nresetting context...");

    // Change the context history and
    let user_message = Message::text("user", "What is 4+4?");
    let messages = vec![system_message, user_message];

    // Use runner to generate new tokens after given context.
    // This handles ChatML transparently via tokenizer.encode_messages
    let context = tokenizer.encode_messages(messages, vec![])?;
    for token in runner.complete(context) {
        print!("{}", tokenizer.decode(vec![token])?);
        let _ = std::io::stdout().flush();
    }

    Ok(())
}
