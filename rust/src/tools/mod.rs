use std::sync::RwLock;

use candle_core::DType;
use candle_nn::{loss, Module, Optimizer};
use mlp::model::{Mlp, MlpConfig};
use tools_trait::CsvLoad;

use crate::frb_generated::StreamSink;

pub mod mlp;
mod tools_trait;

#[derive(Debug, Clone)]
pub struct TrainMessage {
    pub model_name: String,
    pub message: String,
    pub epoch: usize,
    pub loss: f32,
}

impl TrainMessage {
    pub fn default() -> TrainMessage {
        TrainMessage {
            model_name: "".to_string(),
            message: "".to_string(),
            epoch: 0,
            loss: 0.0,
        }
    }
}

pub static TRAIN_MESSAGE_SINK: RwLock<Option<StreamSink<TrainMessage>>> = RwLock::new(None);

pub fn train_a_mlp(csv_path: String) -> anyhow::Result<()> {
    let d = candle_core::Device::cuda_if_available(0)?;
    let vm = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&vm, candle_core::DType::F32, &d);
    let (train_data, train_label) = Mlp::load_csv_without_header(&csv_path)?;
    let train_data = train_data.to_device(&d)?.to_dtype(DType::F32)?;
    let train_label = train_label.to_device(&d)?.to_dtype(DType::F32)?;
    let config = MlpConfig {
        hidden_size: 100,
        input_size: train_data.dims()[1],
        output_size: 1,
    };
    let model = Mlp::new(config, vb)?;
    let param = candle_nn::ParamsAdamW::default();
    let mut adam = candle_nn::AdamW::new(vm.all_vars(), param)?;
    let mut message = TrainMessage::default();
    message.model_name = "mlp".to_string();
    for i in 1..10001 {
        let logits = model.forward(&train_data)?.squeeze(1)?;
        let loss = loss::mse(&logits.to_dtype(DType::F32)?, &train_label)?;
        adam.backward_step(&loss)?;
        if i % 100 == 0 {
            message.loss = loss.to_scalar::<f32>()?;
            message.epoch = i;
            message.message = "".to_string();
            if let Some(sink) = TRAIN_MESSAGE_SINK.read().unwrap().as_ref() {
                let _ = sink.add(message.clone());
            }
        } 
    }

    vm.save("model.bin")?;

    anyhow::Ok(())
}
