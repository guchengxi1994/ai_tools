// use async_channel::Sender;
use candle_core::Device;
use candle_transformers::generation::LogitsProcessor;
use std::marker::PhantomData;

pub struct TextGeneration<T, S, TK> {
    pub model: T,
    pub device: Device,
    pub tokenizer: TK,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub _marker: PhantomData<S>, // 添加 PhantomData<S> 以使用 S
}
