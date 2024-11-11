use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::bigcode::{Config, GPTBigCode},
};
use tokenizers::Tokenizer;

use super::{
    model::{Model, ModelRun},
    text_generation::TextGeneration,
};

impl Model<GPTBigCode> {
    pub fn new(model_path: String) -> Self {
        Model {
            inner: None,
            model_path,
            tokenizer_path: None,
            config_path: None,
        }
    }
}

impl<T> TextGeneration<T, usize, Tokenizer>
where
    T: ModelRun<usize>,
{
    pub fn new(
        model: T,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            device: device.clone(),
            repeat_penalty: 0.0,
            repeat_last_n: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> anyhow::Result<()> {
        use std::io::Write;
        self.model.clear_kv_cache();
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let mut new_tokens = vec![];
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let (context_size, past_len) = if index > 0 {
                (1, tokens.len().saturating_sub(1))
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, past_len)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            let token = self
                .tokenizer
                .decode(&[next_token], true)
                .map_err(anyhow::Error::msg)?;
            print!("{token}");
            std::io::stdout().flush()?;
        }
        let dt = start_gen.elapsed();
        println!(
            "{sample_len} tokens generated ({:.3} token/s)",
            sample_len as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

impl ModelRun<usize> for Model<GPTBigCode> {
    fn forward(
        &mut self,
        xs: &candle_core::Tensor,
        s: usize,
    ) -> anyhow::Result<candle_core::Tensor> {
        match &mut self.inner {
            Some(_i) => anyhow::Ok((*_i).forward(xs, s)?),
            None => anyhow::bail!("model not loaded"),
        }
    }

    fn load(&mut self) -> anyhow::Result<()> {
        let device = Device::cuda_if_available(0)?;
        let model = format!("{}/model.safetensors", self.model_path);
        let token_file_path = format!("{}/tokenizer.json", self.model_path);
        let start = std::time::Instant::now();
        let config_file_path = format!("{}/config.json", self.model_path);
        let config: Config = Config::starcoder_1b();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vec![model], candle_core::DType::F32, &device)?
        };
        println!("[rust-llm] config loaded");
        let model = GPTBigCode::load(vb, config)?;
        self.inner = Some(model);
        self.tokenizer_path = Some(token_file_path);
        self.config_path = Some(config_file_path);

        println!("[rust-llm] loaded the model in {:?}", start.elapsed());

        anyhow::Ok(())
    }

    fn clear_kv_cache(&mut self) {}

    fn get_config_path(&self) -> Option<String> {
        self.config_path.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_bigcode() -> anyhow::Result<()> {
        let device = candle_core::Device::cuda_if_available(0)?;
        println!("[rust-llm] run on device: {:?}", device);
        let mut model =
            Model::<GPTBigCode>::new(r"D:\github_repo\ai_tools\rust\assets\bigcode".to_string());
        model.load()?;
        let tokenizer = Tokenizer::from_file(model.tokenizer_path.clone().unwrap())
            .map_err(anyhow::Error::msg)?;

        let mut pipeline: TextGeneration<Model<GPTBigCode>, usize, Tokenizer> =
            TextGeneration::<Model<GPTBigCode>, usize, Tokenizer>::new(
                model,
                tokenizer,
                128,
                Some(0.7),
                Some(0.9),
                &device,
            );
        let start = std::time::Instant::now();
        let prompt = "
Human: What is the bug in the following code?
```python
   def foo(x: int, y: int) -> int:
       return x / 0 + y
```
Assistant:
        ";
        pipeline.run(prompt, 200)?;
        println!("[rust-llm] total time: {:?}", start.elapsed());

        anyhow::Ok(())
    }
}
