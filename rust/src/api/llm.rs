use flutter_rust_bridge::frb;

use crate::{
    frb_generated::StreamSink,
    llm::{ChatMessages, ChatResponse, CHAT_RESPONSE_SINK},
};

pub fn qwen2_chat(user_prompt: String, system_prompt: Option<String>, model_path: String) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let r = rt.block_on(async {
        crate::llm::qwen2::qwen2_chat(user_prompt, system_prompt, model_path)?;
        anyhow::Ok(())
    });
    match r {
        Ok(_) => {}
        Err(_e) => {
            println!("[rust-llm] Error: {}", _e);
        }
    }
}

pub fn qwen2_prompt_chat(prompt: String, model_path: String) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let r = rt.block_on(async {
        crate::llm::qwen2::qwen2_prompt_chat(prompt, model_path)?;
        anyhow::Ok(())
    });
    match r {
        Ok(_) => {}
        Err(_e) => {
            println!("[rust-llm] Error: {}", _e);
        }
    }
}

#[frb(sync)]
pub fn chat_response_stream(s: StreamSink<ChatResponse>) -> anyhow::Result<()> {
    let mut stream = CHAT_RESPONSE_SINK.write().unwrap();
    *stream = Some(s);
    anyhow::Ok(())
}

#[frb(sync)]
pub fn format_prompt(messages: ChatMessages, system: Option<String>) -> String {
    messages.format(system)
}

#[frb(sync)]
pub fn format_prompt_with_thought_chain(messages: ChatMessages) -> String {
    messages.format_with_thought_chain()
}
