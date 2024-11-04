use std::marker::PhantomData;

use candle_core::Device;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_rwkv_v6::Model as M6;

use candle_transformers::models::rwkv_v5::Config as RwkvConfig;
use candle_transformers::models::rwkv_v5::State as RwkvState;
use candle_transformers::models::rwkv_v5::Tokenizer as RwkvTokenizer;

use super::model::Model;
use super::model::ModelRun;
use super::text_generation::TextGeneration;

impl Model<M6> {
    pub fn new(model_path: String) -> Self {
        Model {
            inner: None,
            model_path,
            tokenizer_path: None,
            config_path: None,
        }
    }
}

impl ModelRun<&mut RwkvState> for Model<M6> {
    fn forward(&mut self, xs: &Tensor, s: &mut RwkvState) -> anyhow::Result<Tensor> {
        // let mut s = s;

        match &mut self.inner {
            Some(_i) => anyhow::Ok((*_i).forward(xs, s)?),
            None => anyhow::bail!("model not loaded"),
        }
    }

    fn get_config_path(&self) -> Option<String> {
        self.config_path.clone()
    }

    fn load(&mut self) -> anyhow::Result<()> {
        let device = Device::cuda_if_available(0)?;
        let model = format!("{}/rwkv-6-world-1b6-q4k.gguf", self.model_path);
        let start = std::time::Instant::now();
        let config_file_path = format!("{}/config.json", self.model_path);
        let token_file_path = format!("{}/rwkv_vocab_v20230424.json", self.model_path);
        let config: RwkvConfig = serde_json::from_slice(&std::fs::read(config_file_path.clone())?)?;
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model, &device)?;
        let model = M6::new(&config, vb)?;
        self.inner = Some(model);
        self.tokenizer_path = Some(token_file_path);
        self.config_path = Some(config_file_path);

        println!("[rust-llm] loaded the model in {:?}", start.elapsed());
        anyhow::Ok(())
    }

    fn clear_kv_cache(&mut self) {
        println!("rwkv clear_kv_cache not implemented")
    }
}

const EOS_TOKEN_ID: u32 = 261;

impl<T> TextGeneration<T, &mut RwkvState, RwkvTokenizer>
where
    T: for<'a> ModelRun<&'a mut RwkvState>,
{
    pub fn new(
        model: T,
        tokenizer: RwkvTokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
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
            callback: None,
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> anyhow::Result<()> {
        use std::io::Write;
        let mut tokens = self.tokenizer.encode(prompt)?;
        let config: RwkvConfig =
            serde_json::from_slice(&std::fs::read(self.model.get_config_path().unwrap())?)?;
        let mut generated_tokens = 0usize;
        let mut state = RwkvState::new(1, &config, &self.device)?;
        let mut next_logits = None;
        for &t in tokens.iter() {
            let input = Tensor::new(&[[t]], &self.device)?;
            let logits = self.model.forward(&input, &mut state)?;
            next_logits = Some(logits);
            print!("{}", self.tokenizer.decode(&[t])?)
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
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
            if next_token == EOS_TOKEN_ID || next_token == 0 {
                break;
            }
            print!("{}", self.tokenizer.decode(&[next_token])?);
            std::io::stdout().flush()?;

            let input = Tensor::new(&[[next_token]], &self.device)?;
            // let state2 = RwkvState::new(1, &config, &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?)
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}
