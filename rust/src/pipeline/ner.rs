use pipeline_rs::{
    node::{AnyMap, NodeRegistration},
    parse_xml,
    pipeline::{get_any, set_any},
};

use crate::llm::{self, ChatMessage, ChatMessages};

const ERA_TEMPLATE : &'static str = "
EXPECTATION (期望): 我希望你能从给定的文本中提取出所有的命名实体，并将它们分类为人物、地点、组织、日期等类别。提取结果需要以JSON格式返回，每种类型的实体应单独列出。

ROLE (角色): 你是一名专业的数据处理专家，擅长从非结构化文本中提取关键信息。你的任务是确保提取的实体准确无误，并按照规定的格式组织数据。

ACTION (行动): 请从以下文本中提取所有命名实体，并将结果以JSON格式返回：
{question}
";

pub struct NERExtractorNode;

impl pipeline_rs::node::Node for NERExtractorNode {
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
            class_name: "NERExtractorNode",
             constructor: || Box::new(NERExtractorNode),
         }
}

pub struct JsonOptimizerNode;

const ERA_JSON_OPTIMIZER_TEMPLATE : &'static str = "
EXPECTATION (期望): 我希望你能验证提供的JSON字符串是否格式正确，并返回一个格式正确的JSON结构。如果输入的JSON字符串已经正确，则直接返回；如果格式错误，则修复并返回正确的JSON结构。

ROLE (角色): 你是一名专业的数据验证和修复专家，负责确保JSON数据的格式正确。你的任务是确保返回的结果是一个格式正确的JSON结构,只需要返回结果，不需要过程。

ACTION (行动): 请验证并提供以下JSON字符串的正确格式：
{question}
";

impl pipeline_rs::node::Node for JsonOptimizerNode {
    fn execute(
        &self,
        context: &mut pipeline_rs::node::AnyMap,
        input_id: String,
        output_id: String,
    ) {
        if let Some(question) = get_any::<String>(&context, &input_id) {
            let chat_message: ChatMessage = ChatMessage {
                role: "user".to_string(),
                content: ERA_JSON_OPTIMIZER_TEMPLATE.replace("{question}", question),
            };
            let chat_messages = ChatMessages {
                0: vec![chat_message],
            };
            let prompt = chat_messages.format(None);
            // set_any(context, &output_id, prompt);
            let r = llm::qwen2::chat_with_cb(prompt, None);
            println!("result {}", r);
            set_any(context, &output_id, r);
        } else {
            panic!("input_id not found")
        }
    }
}

inventory::submit! {
    NodeRegistration {
            class_name: "JsonOptimizerNode",
             constructor: || Box::new(JsonOptimizerNode),
         }
}

pub fn ner_extractor_execute(prompt: String, model_name: Option<String>) -> anyhow::Result<()> {
    let data = include_bytes!(r"ner-pipeline.xml");
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
    fn test_ner_extract() {
        let prompt = "如何学好c++?";
        let model_name = "qwen";
        let result = ner_extractor_execute(prompt.to_string(), Some(model_name.to_string()));
        println!("{:?}", result);
    }
}
