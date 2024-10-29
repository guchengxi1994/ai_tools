use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};


pub struct Model<T> {
    pub inner: Option<T>,
    pub model_path: String,
    pub tokenizer_path: Option<String>,
}

pub trait ModelRun {
    fn forward(&mut self, xs: &Tensor, s: usize) -> anyhow::Result<Tensor>;

    fn load(&mut self) -> anyhow::Result<()>;
}

impl Model<ModelForCausalLM> {
    pub fn new(model_path: String) -> Self {
        Model {
            inner: None,
            model_path,
            tokenizer_path: None,
        }
    }
}

impl ModelRun for Model<candle_transformers::models::qwen2::ModelForCausalLM> {
    fn forward(&mut self, xs: &Tensor, s: usize) -> anyhow::Result<Tensor> {
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
