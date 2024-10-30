#[allow(unused_imports, dead_code)]
mod mlp_test {
    use super::super::macros::push_fields;
    use std::{
        fs::File,
        io::{self, Write},
    };

    use candle_core::{DType, Device, Tensor, D};
    use candle_nn::{linear, loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};

    // https://huggingface.github.io/candle/training/simplified.html
    pub struct MLPCommon {
        pub input_size: usize,
        pub hidden_sizes: Vec<usize>,
        pub dropout: Vec<f64>,
        pub bn: Vec<bool>,
        pub output_size: usize,
    }

    #[derive(Clone)]
    pub struct DataSet {
        pub train_votes: Tensor,
        pub train_results: Tensor,
        pub test_votes: Tensor,
        pub test_results: Tensor,
    }

    #[derive(Debug, serde::Deserialize)]
    struct Record {
        pub p1: f32,
        pub p2: f32,
        pub p3: f32,
        pub p4: f32,
        pub p5: f32,
        pub p6: f32,
        pub p7: f32,
        pub p8: f32,
        pub p9: f32,
        pub p10: f32,
        pub p11: f32,
        pub p12: f32,
        pub p13: f32,
        pub p14: f32,
        pub p15: f32,
        pub p16: f32,
        pub p17: f32,
        pub p18: f32,
        pub p19: f32,
        pub p20: f32,
        pub p21: f32,
        pub p22: f32,
        pub p23: f32,
        pub p24: f32,
        pub p25: f32,
        pub p26: f32,
        pub p27: f32,
        pub p28: f32,
        pub p29: f32,
        pub p30: f32,
        pub p31: f32,
        pub p32: f32,
        pub p33: f32,
        pub p34: f32,
        pub p35: f32,
        pub p36: f32,
        pub p37: f32,
        pub score: f32,
    }

    pub struct Records(Vec<Record>);

    impl Records {
        pub fn from_file(path: &str) -> anyhow::Result<Self> {
            let file = File::open(path)?;
            let mut rdr = csv::Reader::from_reader(file);
            let mut records: Vec<Record> = vec![];
            for result in rdr.deserialize() {
                let record: Record = result?;
                records.push(record);
            }
            Ok(Self(records))
        }

        pub fn to_dataset(self, rate: f32, device: Device) -> anyhow::Result<DataSet> {
            let train_votes;
            let train_results;
            let test_votes;
            let test_results;
            let mut v: Vec<Vec<f32>> = vec![];
            let mut scores: Vec<f32> = vec![];
            for record in self.0 {
                let mut row: Vec<f32> = vec![];
                push_fields!(
                    row, record, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
                    p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31,
                    p32, p33, p34, p35, p36, p37
                );
                v.push(row);
                scores.push(record.score);
            }

            let count = (rate * v.len() as f32) as usize;
            let train_v = v[0..count].to_vec().concat();
            let train_r = scores[0..count].to_vec();
            train_votes = Tensor::from_vec(train_v, (count, 37), &device)?;
            train_results = Tensor::from_vec(train_r, (count, 1), &device)?.to_dtype(DType::F32)?;

            let test_v = v[count..].to_vec().concat();
            test_votes = Tensor::from_vec(test_v, (v.len() - count, 37), &device)?;
            test_results =
                Tensor::from_vec(scores[count..].to_vec(), (scores.len() - count, 1), &device)?
                    .to_dtype(DType::F32)?;

            Ok(DataSet {
                train_votes,
                train_results,
                test_votes,
                test_results,
            })
        }
    }

    struct MultiLevelPerceptron {
        ln1: Linear,
        ln2: Linear,
        ln3: Linear,
    }

    impl MultiLevelPerceptron {
        fn new(vs: VarBuilder) -> anyhow::Result<Self> {
            let ln1 = candle_nn::linear(37, 100, vs.pp("ln1"))?;
            let ln2 = candle_nn::linear(100, 50, vs.pp("ln2"))?;
            let ln3 = candle_nn::linear(50, 1, vs.pp("ln3"))?;
            Ok(Self { ln1, ln2, ln3 })
        }

        fn forward(&self, xs: &Tensor) -> anyhow::Result<Tensor> {
            let xs = self.ln1.forward(xs)?;
            let xs = xs.relu()?;
            let xs = ops::dropout(&xs, 0.5)?;
            let xs = self.ln2.forward(&xs)?;
            let xs = xs.relu()?;
            let xs = ops::dropout(&xs, 0.5)?;
            Ok(self.ln3.forward(&xs)?)
        }
    }

    impl MLPCommon {
        pub fn forward(&self, x: &Tensor, device: Device) -> candle_core::Result<Tensor> {
            let mut input_size = self.input_size;
            let output_size = self.output_size;
            if self.hidden_sizes.is_empty() {
                let weight = Tensor::randn(0f32, 1.0, (output_size, input_size), &device)?;
                let bias = Tensor::randn(0f32, 1.0, (output_size,), &device)?;
                let l: Linear = Linear::new(weight, Some(bias));
                return l.forward(x);
            } else {
                let mut input = x.copy()?;

                for (index, value) in self.hidden_sizes.iter().enumerate() {
                    println!(" {} -> {}", input_size, value);
                    let weight = Tensor::randn(0f32, 1.0, (value.clone(), input_size), &device)?;
                    let bias = Tensor::randn(0f32, 1.0, (value.clone(),), &device)?;
                    let l: Linear = Linear::new(weight, Some(bias));
                    input = l.forward(&input)?;
                    input = input.relu()?;
                    input_size = value.clone();
                }

                let weight = Tensor::randn(0f32, 1.0, (output_size, input_size), &device)?;
                let bias = Tensor::randn(0f32, 1.0, (output_size,), &device)?;
                let l: Linear = Linear::new(weight, Some(bias));
                input = l.forward(&input)?;
                return Ok(input);
            }
        }
    }

    pub fn train(m: DataSet, dev: Device) -> anyhow::Result<()> {
        let train_results = m.train_results.to_device(&dev)?;
        let train_votes = m.train_votes.to_device(&dev)?;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = MultiLevelPerceptron::new(vs.clone())?;
        // let mut sgd = candle_nn::SGD::new(varmap.all_vars(), 0.05)?;
        let param = candle_nn::ParamsAdamW::default();
        let mut adam = candle_nn::AdamW::new(varmap.all_vars(), param)?;
        let test_votes = m.test_votes.to_device(&dev)?;
        let test_results = m.test_results.to_device(&dev)?;
        // let mut final_accuracy: f32 = 0.0;

        for epoch in 1..100_000 {
            let logits = model.forward(&train_votes)?;
            let loss = loss::mse(&logits.to_dtype(DType::F32)?, &train_results)?;
            let test_logits = model.forward(&test_votes)?;
            let test_losses = loss::mse(&test_logits.to_dtype(DType::F32)?, &test_results)?;
            adam.backward_step(&loss)?;

            if epoch % 1000 == 0 {
                println!("epoch: {}, loss: {}", epoch, loss);
                io::stdout().flush().unwrap();
                println!("test_loss: {}", test_losses);
                io::stdout().flush().unwrap();
            }
        }

        varmap.save("model.bin")?;
        anyhow::Ok(())
    }

    #[test]
    pub fn test_load_model() -> anyhow::Result<()> {
        let dev = Device::cuda_if_available(0)?;
        let mut varmap = VarMap::new();
        varmap.load("model.bin")?;
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = MultiLevelPerceptron::new(vs)?;
        let test_data = Tensor::randn(0f32, 20f32, (1, 37), &dev)?;
        println!("{:?}", model.forward(&test_data)?.to_vec2::<f32>()?);

        anyhow::Ok(())
    }

    #[test]
    fn m_test() -> anyhow::Result<()> {
        let dev = Device::cuda_if_available(0)?;

        let records = Records::from_file("assets/data.csv")?;
        let m = records.to_dataset(0.8, dev.clone())?;

        train(m.clone(), dev)?;
        anyhow::Ok(())
    }

    #[test]
    fn example() -> anyhow::Result<()> {
        let mut rdr = csv::Reader::from_reader(File::open("assets/data.csv")?);
        for result in rdr.deserialize() {
            // Notice that we need to provide a type hint for automatic
            // deserialization.
            let record: Record = result?;
            println!("{:?}", record);
        }
        Ok(())
    }
}
