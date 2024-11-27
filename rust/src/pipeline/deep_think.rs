use pipeline_rs::{
    node::{AnyMap, NodeRegistration},
    parse_xml,
    pipeline::{get_any, set_any},
};

use crate::llm::{self, ChatMessage, ChatMessages};

pub struct RewriteQuestionNode;

const ERA_TEMPLATE : &'static str = "
EXPECTATION (期望): 我希望你能将提供的原始问题进行重写，使得新问题不仅更加清晰具体，而且能够激发更多的思考角度。重写后的问题应该能够帮助回答者更深入地探讨主题，同时避免过于狭窄或封闭的答案。

ROLE (角色): 你是一位专业的编辑，专门负责优化问题以提高讨论的质量和深度。你的任务是确保每个问题都能够引发有意义的对话，促进知识的分享与探索。

ACTION (行动): 请接收以下原始问题：“{question}”对其进行重写，使其成为既能引导深入探讨又能启发多方面思考的新问题。只需要输出结果，不需要过程。
";

impl pipeline_rs::node::Node for RewriteQuestionNode {
    fn execute(
        &self,
        context: &mut pipeline_rs::node::AnyMap,
        input_id: String,
        output_id: String,
    ) {
        if let Some(question) = get_any::<String>(&context, &input_id) {
            let chat_message: ChatMessage = ChatMessage {
                role: "user".to_string(),
                content: ERA_TEMPLATE.replace("{question}", question),
            };
            let chat_messages = ChatMessages {
                0: vec![chat_message],
            };
            let prompt = chat_messages.format(None);
            // set_any(context, &output_id, prompt);
            let r = llm::qwen2::chat_with_cb(prompt, None);
            println!("rewrite_question {}", r);
            set_any(context, &output_id, r);
        } else {
            panic!("input_id not found")
        }
    }
}

inventory::submit! {
    NodeRegistration {
            class_name: "RewriteQuestionNode",
             constructor: || Box::new(RewriteQuestionNode),
         }
}

pub struct RethinkNode;

const RETHINK_ERA_TEMPLATE : &'static str = "
EXPECTATION (期望): 我希望结合输入的问题和输出的答案，判断答案的质量。

ROLE (角色): 你的任务是确保每个问题都能够给出公平公正的回答。

ACTION (行动): 请接收以下原始问题：“{question}” 和输出的答案：“{answer}”。请判断其相关性以及评估回答的质量。只需要输出结果，不需要过程。
";

impl pipeline_rs::node::Node for RethinkNode {
    fn execute(
        &self,
        context: &mut pipeline_rs::node::AnyMap,
        input_id: String,
        output_id: String,
    ) {
        let question = get_any::<String>(&context, "prompt").unwrap();
        let answer = get_any::<String>(&context, "result").unwrap();

        let p = RETHINK_ERA_TEMPLATE
            .replace("{question}", question)
            .replace("{answer}", answer);

        let chat_message: ChatMessage = ChatMessage {
            role: "user".to_string(),
            content: p,
        };
        let chat_message_intl: ChatMessage = ChatMessage {
            role: "user".to_string(),
            content: "答案请使用中文。".to_string(),
        };

        let chat_messages = ChatMessages {
            0: vec![chat_message, chat_message_intl],
        };
        let prompt = chat_messages.format(None);
        let r = llm::qwen2::chat_with_cb(prompt, None);
        println!("rethink {}", r);
        set_any(context, &output_id, r);
    }
}

inventory::submit! {
    NodeRegistration {
            class_name: "RethinkNode",
             constructor: || Box::new(RethinkNode),
         }
}

pub fn deep_think_execute(prompt: String, model_name: Option<String>) -> anyhow::Result<()> {
    let data = include_bytes!(r"rewrite-question-pipeline.xml");
    let pipeline = parse_xml(std::str::from_utf8(data)?)?;
    for action in &pipeline.actions {
        println!("{:?}", action);
    }
    let mut context = AnyMap::new();
    context.insert("prompt".to_string(), Box::new(prompt));
    context.insert(
        "model-name".to_string(),
        Box::new(model_name.unwrap_or("qwen".to_string())),
    );
    pipeline.execute_with_input(
        &mut context,
        Some(|x| {
            println!("error {:?}", x);
        }),
        None,
    );
    anyhow::Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deep_think() {
        let prompt = "如何学好c++?";
        let model_name = "qwen";
        let result = deep_think_execute(prompt.to_string(), Some(model_name.to_string()));
        println!("{:?}", result);
    }
}
