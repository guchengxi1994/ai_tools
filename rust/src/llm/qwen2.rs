use std::marker::PhantomData;
use std::sync::RwLock;

use super::{model::ModelRun, token_output_stream::TokenOutputStream};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::Config as Qwen2Config;
use candle_transformers::models::qwen2::ModelForCausalLM;
use futures::future::BoxFuture;
use once_cell::sync::Lazy;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;

use crate::constant::DEFAULT_SYSTEM_ROLE;
use crate::llm::ChatResponse;
use crate::llm::CHAT_RESPONSE_SINK;

pub static QWEN_MODEL: Lazy<
    RwLock<
        Option<
            TextGeneration<
                Model<ModelForCausalLM>,
                usize,
                crate::llm::token_output_stream::TokenOutputStream,
            >,
        >,
    >,
> = Lazy::new(|| RwLock::new(None));

pub fn clear_all_models() {
    {
        let mut qwen_model = QWEN_MODEL.write().unwrap();
        *qwen_model = None;
    }
}

use super::{model::Model, text_generation::TextGeneration, BASE_TEMPLATE};

impl Model<ModelForCausalLM> {
    pub fn new(model_path: String) -> Self {
        Model {
            inner: None,
            model_path,
            tokenizer_path: None,
            config_path: None,
        }
    }
}

impl ModelRun<usize> for Model<candle_transformers::models::qwen2::ModelForCausalLM> {
    fn forward(&mut self, xs: &Tensor, s: usize) -> anyhow::Result<Tensor> {
        match &mut self.inner {
            Some(_i) => anyhow::Ok((*_i).forward(xs, s)?),
            None => anyhow::bail!("model not loaded"),
        }
    }

    fn get_config_path(&self) -> Option<String> {
        self.config_path.clone()
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.inner {
            Some(_i) => _i.clear_kv_cache(),
            None => {}
        }
    }

    fn load(&mut self) -> anyhow::Result<()> {
        let device = Device::cuda_if_available(0)?;
        let model = format!("{}/model.safetensors", self.model_path);
        let token_file_path = format!("{}/tokenizer.json", self.model_path);
        let start = std::time::Instant::now();
        let config_file_path = format!("{}/config.json", self.model_path);
        let config: Qwen2Config = serde_json::from_slice(&std::fs::read(config_file_path)?)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vec![model], candle_core::DType::F32, &device)?
        };
        println!("[rust-llm] config loaded");
        let model = ModelForCausalLM::new(&config, vb)?;
        self.inner = Some(model);
        self.tokenizer_path = Some(token_file_path);

        println!("[rust-llm] loaded the model in {:?}", start.elapsed());

        anyhow::Ok(())
    }
}

impl<T> TextGeneration<T, usize, TokenOutputStream>
where
    T: ModelRun<usize>,
{
    pub fn set_callback(
        &mut self,
        callback: Option<
            Box<dyn Fn(String, Sender<String>) -> BoxFuture<'static, ()> + Send + Sync>,
        >,
    ) {
        self.callback = callback;
    }

    pub fn new(
        model: T,
        tokenizer: TokenOutputStream,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
        callback: Option<
            Box<dyn Fn(String, Sender<String>) -> BoxFuture<'static, ()> + Send + Sync>,
        >,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            _marker: PhantomData, // 初始化 PhantomData
            callback,
        }
    }

    pub fn run(
        &mut self,
        prompt: &str,
        sample_len: usize,
        tx: Option<Sender<String>>,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        self.model.clear_kv_cache();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let start_gen = std::time::Instant::now();
        let mut chat_response = ChatResponse::new();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(candle_core::DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                if self.callback.is_some() {
                    let _ = (self.callback.as_ref().unwrap())(t.to_string(), tx.clone().unwrap());
                } else {
                    chat_response.set_content(t);
                    if let Some(s) = CHAT_RESPONSE_SINK.read().unwrap().as_ref() {
                        let _ = s.add(chat_response.clone());
                    }
                }

                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        if !self.callback.is_some() {
            chat_response.set_content("".to_string());
            chat_response.set_done(true);
            chat_response.set_tps(generated_tokens as f64 / dt.as_secs_f64());
            chat_response.set_stage("done".to_string());
            chat_response.set_token_generated(generated_tokens);
            if let Some(s) = CHAT_RESPONSE_SINK.read().unwrap().as_ref() {
                let _ = s.add(chat_response.clone());
            }
        }
        Ok(())
    }
}

/// expose to flutter
pub fn qwen2_prompt_chat(p: String, model_path: String) -> anyhow::Result<()> {
    {
        let mut global_model = QWEN_MODEL.write().unwrap();
        if !global_model.is_none() {
            println!("[rust-llm] use global model");
            let start = std::time::Instant::now();
            global_model.as_mut().unwrap().run(&p, 1024, None)?;
            println!("end in {:?}", start.elapsed());
            return Ok(());
        }
    }

    let device = candle_core::Device::cuda_if_available(0)?;
    println!("[rust-llm] run on device: {:?}", device);

    let mut model = Model::<ModelForCausalLM>::new(model_path);
    model.load()?;

    let tokenizer =
        Tokenizer::from_file(&model.tokenizer_path.clone().unwrap()).map_err(anyhow::Error::msg)?;

    let mut pipeline = TextGeneration::<
        Model<ModelForCausalLM>,
        usize,
        crate::llm::token_output_stream::TokenOutputStream,
    >::new(
        model,
        TokenOutputStream::new(tokenizer),
        128,
        Some(0.7),
        Some(0.9),
        1.25,
        64,
        &device,
        None,
    );

    let start = std::time::Instant::now();

    pipeline.run(&p, 1024, None)?;

    {
        let mut global_model = QWEN_MODEL.write().unwrap();
        *global_model = Some(pipeline);
    }
    println!("end in {:?}", start.elapsed());
    anyhow::Ok(())
}

/// expose to flutter
pub fn qwen2_chat(
    user_prompt: String,
    system_prompt: Option<String>,
    model_path: String,
) -> anyhow::Result<()> {
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
    {
        let mut global_model = QWEN_MODEL.write().unwrap();
        if !global_model.is_none() {
            println!("[rust-llm] use global model");
            let start = std::time::Instant::now();
            global_model.as_mut().unwrap().run(&prompt, 1024, None)?;
            println!("end in {:?}", start.elapsed());
            return Ok(());
        }
    }

    let device = candle_core::Device::cuda_if_available(0)?;
    println!("[rust-llm] run on device: {:?}", device);

    let mut model = Model::<ModelForCausalLM>::new(model_path);
    model.load()?;

    let tokenizer =
        Tokenizer::from_file(&model.tokenizer_path.clone().unwrap()).map_err(anyhow::Error::msg)?;

    let mut pipeline = TextGeneration::<
        Model<ModelForCausalLM>,
        usize,
        crate::llm::token_output_stream::TokenOutputStream,
    >::new(
        model,
        TokenOutputStream::new(tokenizer),
        128,
        Some(0.7),
        Some(0.9),
        1.25,
        64,
        &device,
        None,
    );

    let start = std::time::Instant::now();

    pipeline.run(&prompt, 1024, None)?;

    {
        let mut global_model = QWEN_MODEL.write().unwrap();
        *global_model = Some(pipeline);
    }
    println!("end in {:?}", start.elapsed());
    anyhow::Ok(())
}
