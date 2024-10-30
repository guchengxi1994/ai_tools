use std::sync::RwLock;

use uuid::Uuid;

use crate::frb_generated::StreamSink;

pub mod model;
pub mod qwen2;
pub mod text_generation;
pub mod token_output_stream;

pub static CHAT_RESPONSE_SINK: RwLock<Option<StreamSink<ChatResponse>>> = RwLock::new(None);

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub done: bool,
    pub stage: String,
    pub uuid: String,
}

impl ChatResponse {
    pub fn new() -> Self {
        let id = Uuid::new_v4();

        Self {
            content: "".to_string(),
            done: false,
            stage: "".to_string(),
            uuid: id.to_string(),
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
