use candle_transformers::models::qwen2::ModelForCausalLM;
use tokenizers::Tokenizer;

use crate::constant::DEFAULT_SYSTEM_ROLE;

use super::{
    model::{Model, ModelRun},
    text_generation::TextGeneration,
    BASE_TEMPLATE,
};

pub fn qwen2_chat(
    user_prompt: String,
    system_prompt: Option<String>,
    model_path: String,
) -> anyhow::Result<()> {
    let device = candle_core::Device::cuda_if_available(0)?;
    println!("[rust-llm] run on device: {:?}", device);

    let prompt;
    match system_prompt {
        Some(_s) => {
            prompt = BASE_TEMPLATE
                .replace("{system}", &_s)
                .replace("{user}", &user_prompt);
        }
        None => {
            prompt = BASE_TEMPLATE
                .replace("{system}", DEFAULT_SYSTEM_ROLE)
                .replace("{user}", &user_prompt);
        }
    }

    let mut model = Model::<ModelForCausalLM>::new(model_path);
    model.load()?;

    let tokenizer =
        Tokenizer::from_file(&model.tokenizer_path.clone().unwrap()).map_err(anyhow::Error::msg)?;

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        128,
        Some(0.7),
        Some(0.9),
        1.25,
        64,
        &device,
    );

    let start = std::time::Instant::now();

    pipeline.run(&prompt, 1024)?;

    println!("end in {:?}", start.elapsed());

    anyhow::Ok(())
}
