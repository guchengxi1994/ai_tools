// use async_channel::Sender;
use candle_core::Device;
use candle_transformers::generation::LogitsProcessor;
use futures::future::BoxFuture;
use std::marker::PhantomData;
use tokio::sync::mpsc::Sender;

pub struct TextGeneration<T, S, TK> {
    pub model: T,
    pub device: Device,
    pub tokenizer: TK,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub _marker: PhantomData<S>, // 添加 PhantomData<S> 以使用 S
    pub callback:
        Option<Box<dyn Fn(String, Sender<String>) -> BoxFuture<'static, ()> + Send + Sync>>,
}
