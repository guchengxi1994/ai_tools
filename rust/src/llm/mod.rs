use std::sync::RwLock;

use uuid::Uuid;

use crate::{
    constant::{DEFAULT_SYSTEM_ROLE, THOUGHT_CHAIN_SYSTEM_ROLE},
    frb_generated::StreamSink,
};

pub mod model;
pub mod qwen2;
pub mod rwkv;
pub mod text_generation;
pub mod token_output_stream;

pub static CHAT_RESPONSE_SINK: RwLock<Option<StreamSink<ChatResponse>>> = RwLock::new(None);

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub done: bool,
    pub stage: String,
    pub uuid: String,
    pub tps: f64,
    pub token_generated: usize,
}

impl ChatResponse {
    pub fn new() -> Self {
        let id = Uuid::new_v4();

        Self {
            content: "".to_string(),
            done: false,
            stage: "".to_string(),
            uuid: id.to_string(),
            tps: 0.0,
            token_generated: 0,
        }
    }

    pub fn set_content(&mut self, content: String) {
        self.content = content;
    }

    pub fn set_done(&mut self, done: bool) {
        self.done = done;
    }

    pub fn set_stage(&mut self, stage: String) {
        self.stage = stage;
    }

    pub fn set_uuid(&mut self, uuid: String) {
        self.uuid = uuid;
    }

    pub fn set_tps(&mut self, tps: f64) {
        self.tps = tps;
    }

    pub fn set_token_generated(&mut self, token_generated: usize) {
        self.token_generated = token_generated;
    }
}

pub const BASE_TEMPLATE: &str = "
<|im_start|>system
{system}
<|im_end|>

<|im_start|>user
{user}
<|im_end|>

<|im_start|>assistant
";

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct ChatMessages(pub Vec<ChatMessage>);

impl ChatMessages {
    pub fn format(&self, system: Option<String>) -> String {
        let mut result = String::new();
        if let Some(system) = system {
            result += &format!("<|im_start|>system\n{}\n<|im_end|>\n", system);
        } else {
            result += &format!("<|im_start|>system\n{}\n<|im_end|>\n", DEFAULT_SYSTEM_ROLE);
        }

        for message in self.0.iter() {
            result += &format!(
                "<|im_start|>{}\n{}\n<|im_end|>\n",
                message.role, message.content
            );
        }
        result += "<|im_start|>assistant\n";

        result
    }

    pub fn format_with_thought_chain(&self) -> String {
        let mut result = String::new();
        result += &format!(
            "<|im_start|>system\n{}\n<|im_end|>\n",
            THOUGHT_CHAIN_SYSTEM_ROLE
        );

        for message in self.0.iter() {
            result += &format!(
                "<|im_start|>{}\n{}\n<|im_end|>\n",
                message.role, message.content
            );
        }
        result += "<|im_start|>assistant\n现在我将一步步思考，从分析问题开始并将问题分解\n";

        result
    }
}

pub async fn clear_all_models_async() {
    qwen2::clear_all_models_async().await;
}
