use pipeline_rs::{
    node::NodeRegistration,
    pipeline::{get_any, set_any},
};

use crate::{
    llm::{self, ChatMessage, ChatMessages},
    web_server,
};

pub struct CheckModelRunningNode;

impl pipeline_rs::node::Node for CheckModelRunningNode {
    fn execute(
        &self,
        context: &mut pipeline_rs::node::AnyMap,
        input_id: String,
        _output_id: String,
    ) {
        if let Some(input) = get_any::<String>(&context, &input_id) {
            if input == "qwen" {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let r = rt.block_on(async {
                    return web_server::check_llm_model_loaded().await;
                });
                if !r {
                    panic!("model not load")
                }
            }
        } else {
            panic!("model not found");
        }
    }
}

inventory::submit! {
    NodeRegistration {
            class_name: "CheckModelRunningNode",
             constructor: || Box::new(CheckModelRunningNode),
         }
}

pub struct CheckOrRunModelNode;

impl pipeline_rs::node::Node for CheckOrRunModelNode {
    fn execute(
        &self,
        context: &mut pipeline_rs::node::AnyMap,
        input_id: String,
        _output_id: String,
    ) {
        if let Some(input) = get_any::<String>(&context, &input_id) {
            if input == "qwen" {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let r = web_server::check_llm_model_loaded().await;
                    if !r {
                        let _ = web_server::load_model(None).await;
                    }
                });
            }
        } else {
            panic!("model not found");
        }
    }
}

inventory::submit! {
    NodeRegistration {
            class_name: "CheckOrRunModelNode",
             constructor: || Box::new(CheckOrRunModelNode),
         }
}

pub struct AnswerQuestionNode;

impl pipeline_rs::node::Node for AnswerQuestionNode {
    fn execute(
        &self,
        context: &mut pipeline_rs::node::AnyMap,
        input_id: String,
        output_id: String,
    ) {
        if let Some(question) = get_any::<String>(&context, &input_id) {
            let chat_message: ChatMessage = ChatMessage {
                role: "user".to_string(),
                content: question.to_string(),
            };
            let chat_message_intl: ChatMessage = ChatMessage {
                role: "user".to_string(),
                content: "答案请使用中文。".to_string(),
            };
            let chat_messages = ChatMessages {
                0: vec![chat_message, chat_message_intl],
            };
            let prompt = chat_messages.format(None);
            // set_any(context, &output_id, prompt);
            let r = llm::qwen2::chat_with_cb(prompt, None);
            println!("result {}", r);
            set_any(context, &output_id, r);
        } else {
            panic!("input_id not found");
        }
    }
}

inventory::submit! {
    NodeRegistration {
            class_name: "AnswerQuestionNode",
             constructor: || Box::new(AnswerQuestionNode),
         }
}
