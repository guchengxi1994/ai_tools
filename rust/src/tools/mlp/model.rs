use std::fs::File;

use candle_core::Tensor;
use candle_nn::{linear, ops, Linear, Module, VarBuilder};

use crate::tools::tools_trait::CsvLoad;

pub struct Mlp {
    pub linear1: Linear,
    pub linear2: Linear,
}

pub struct MlpConfig {
    pub hidden_size: usize,
    pub input_size: usize,
    pub output_size: usize,
}

impl MlpConfig {
    pub fn default() -> Self {
        MlpConfig {
            hidden_size: 100,
            input_size: 100,
            output_size: 1,
        }
    }
}

impl Mlp {
    pub fn new(config: MlpConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let linear1 = linear(config.input_size, config.hidden_size, vb.pp("linear1"))?;
        let linear2 = linear(config.hidden_size, config.output_size, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let xs = self.linear1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = ops::dropout(&xs, 0.5)?;
        self.linear2.forward(&xs)
    }
}

impl CsvLoad for Mlp {
    fn load_csv(path: &str) -> anyhow::Result<(candle_core::Tensor, candle_core::Tensor)> {
        load_csv_impl(path, 0)
    }

    fn load_csv_without_header(
        path: &str,
    ) -> anyhow::Result<(candle_core::Tensor, candle_core::Tensor)> {
        load_csv_impl(path, 1)
    }
}

fn load_csv_impl(path: &str, start_index: usize) -> anyhow::Result<(Tensor, Tensor)> {
    let file = File::open(path)?;
    let mut reader = csv::Reader::from_reader(file);

    let mut ys = Vec::new();
    let mut xs = Vec::new();
    let mut index = 0;

    for result in reader.records() {
        if index < start_index {
            index += 1;
            continue;
        }
        if let Ok(record) = result {
            let mut v: Vec<f64> = Vec::new();
            record
                .iter()
                .for_each(|x| v.push(x.parse::<f64>().unwrap_or(0f64)));

            ys.push(v.pop().unwrap());
            xs.push(Tensor::from_vec(
                v.clone(),
                &[v.len()],
                &candle_core::Device::Cpu,
            )?);
        }
    }
    return Ok((
        Tensor::stack(&xs, 0)?,
        Tensor::from_vec(ys.clone(), &[ys.len()], &candle_core::Device::Cpu)?,
    ));
}

#[cfg(test)]
mod mlp_test {
    use candle_core::Device;
    use candle_nn::{loss, Module};

    use crate::tools::tools_trait::CsvLoad;

    #[test]
    fn load_file() -> anyhow::Result<()> {
        let r = super::load_csv_impl(r"assets\data.csv", 1)?;
        println!("{:?}", r.0.shape());
        println!("{:?}", r.1.shape());
        anyhow::Ok(())
    }

    #[test]
    fn test_train() -> anyhow::Result<()> {
        let d = Device::cuda_if_available(0)?;
        let mut cfg = super::MlpConfig::default();
        cfg.input_size = 37;
        cfg.hidden_size = 100;
        cfg.output_size = 1;

        let (xs, ys) = super::Mlp::load_csv_without_header(r"assets\data.csv")?;
        let xs = xs.to_device(&d)?.to_dtype(candle_core::DType::F32)?;
        let ys = ys.to_device(&d)?.to_dtype(candle_core::DType::F32)?;

        let dataset = candle_dataset_loader::dataset::Dataset {
            train_data: xs,
            train_labels: ys,
            test_data: None,
            test_labels: None,
            batch_size: 1000,
        };

        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, candle_core::DType::F32, &d);
        let model = super::Mlp::new(cfg, vb)?;
        // let loss = model.forward(&xs)?.mse_loss(&ys)?;

        let _ = dataset.into_iter().take(30).try_for_each(|(x, y)| {
            let result = model.forward(&x)?.squeeze(1)?;
            let loss = loss::mse(
                &result.to_dtype(candle_core::DType::F32)?.to_device(&d)?,
                &y,
            )?;
            println!("loss ===> {:?}", loss);
            Ok::<(), Box<dyn std::error::Error>>(())
        });

        anyhow::Ok(())
    }
}
