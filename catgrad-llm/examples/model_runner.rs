use catgrad::prelude::Dtype;
use catgrad_llm::run::ModelEngine;
use catgrad_llm::types::Message;
use catgrad_llm::types::openai::ChatMessage;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ModelEngine::new("Qwen/Qwen3-0.6B", true, Dtype::F32)?;

    // Make some message context
    let system_message = Message::openai(ChatMessage::system("You are a helpful chat assistant"));
    let user_message = Message::openai(ChatMessage::user("What is 2+2?"));
    let prompt = engine.prepare_messages(&[system_message.clone(), user_message])?;
    engine.generate_from_prepared(&prompt, 128, |delta| {
        print!("{delta}");
        let _ = std::io::stdout().flush();
        Ok(())
    })?;

    ////////////////////////////////////////////////////////////////////////////
    // Change user_message to modify context history

    println!("\nresetting context...");

    // Change the context history and
    let user_message = Message::openai(ChatMessage::user("What is 4+4?"));
    let prompt = engine.prepare_messages(&[system_message, user_message])?;
    engine.generate_from_prepared(&prompt, 128, |delta| {
        print!("{delta}");
        let _ = std::io::stdout().flush();
        Ok(())
    })?;

    Ok(())
}
