use candle_core::Tensor;

pub struct Model<T> {
    pub inner: Option<T>,
    pub model_path: String,
    pub tokenizer_path: Option<String>,
    pub config_path: Option<String>,
}

pub trait ModelRun<S>
where
    S: ?Sized,
{
    fn forward(&mut self, xs: &Tensor, s: S) -> anyhow::Result<Tensor>;

    fn load(&mut self) -> anyhow::Result<()>;

    fn clear_kv_cache(&mut self);

    fn get_config_path(&self) -> Option<String>;
}
