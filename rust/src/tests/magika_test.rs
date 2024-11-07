use candle_core::{Device, Tensor};
use candle_nn::layer_norm;
use candle_nn::{
    ops::{self},
    LayerNormConfig, Module, VarBuilder, VarMap,
};

pub struct Magika {
    dense1: candle_nn::Linear,
    dense2: candle_nn::Linear,
    dense3: candle_nn::Linear,
    dense4: candle_nn::Linear,
    layer_norm1: layer_norm::LayerNorm,
    layer_norm2: layer_norm::LayerNorm,
}

impl Magika {
    pub fn new(vb: VarBuilder, output_size: usize) -> anyhow::Result<Self> {
        let dense1 = candle_nn::linear(257, 128, vb.pp("dense1"))?;
        let dense2 = candle_nn::linear(512, 256, vb.pp("dense2"))?;
        let dense3 = candle_nn::linear(256, 256, vb.pp("dense3"))?;
        let dense4 = candle_nn::linear(256, output_size, vb.pp("dense4"))?;
        let cfg: LayerNormConfig = LayerNormConfig::default();
        let layer_norm1 = candle_nn::layer_norm(512, cfg, vb.pp("layer_norm1"))?;
        let layer_norm2 = candle_nn::layer_norm(256, cfg, vb.pp("layer_norm2"))?;

        Ok(Self {
            dense1,
            dense2,
            dense3,
            dense4,
            layer_norm1,
            layer_norm2,
        })
    }
}

impl Module for Magika {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // one hot encoding
        let dims = xs.dims();
        let batch_size = dims[0];
        let mut batch = Vec::<Tensor>::new();
        for i in 0..batch_size {
            let x = xs.narrow(0, i, 1)?.squeeze(0)?;
            let x = one_hot(&x, 257, candle_core::DType::U8).unwrap();
            let x = x.to_dtype(candle_core::DType::F32)?;
            batch.push(x);
        }
        let xs = Tensor::stack(&batch, 0)?;
        println!("xs shape: {:?}", xs.shape());
        let xs = self.dense1.forward(&xs)?;
        println!("xs shape: {:?}", xs.shape());
        // [TODO]: shoule be `SpatialDropout`
        let xs = ops::dropout(&xs, 0.5)?;
        let xs = xs.reshape(&[xs.dims()[0], 384, 512])?;

        let xs = self.layer_norm1.forward(&xs)?;
        let xs = ops::dropout(&xs, 0.5)?;
        let xs = self.dense2.forward(&xs)?;
        let xs = self.dense3.forward(&xs)?;
        println!("xs shape: {:?}", xs.shape());
        // GlobalMaxPooling1D
        let xs = xs.max(1)?;
        println!("xs shape: {:?}", xs.shape());
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = ops::dropout(&xs, 0.5)?;
        let xs = self.dense4.forward(&xs)?;

        Ok(xs)
    }
}

fn one_hot(
    labels: &Tensor,
    num_classes: usize,
    r#type: candle_core::DType,
) -> anyhow::Result<Tensor> {
    let shape = labels.shape();
    let one_hot = Tensor::zeros(&[shape.dims()[0], num_classes], r#type, labels.device())?;
    let mut v: Vec<Tensor> = vec![];
    for i in 0..shape.dims()[0] {
        let l = labels.get(i)?;
        let l = l.to_scalar::<u8>()?;
        let l = l as usize;
        let mut t = Tensor::zeros(&[num_classes], r#type, labels.device())?;
        t = t.slice_assign(&[l..l + 1], &Tensor::ones(&[1], r#type, labels.device())?)?;
        v.push(t);
    }

    Ok(Tensor::stack(&v, 0)?)
}

#[test]
fn one_hot_test() -> anyhow::Result<()> {
    let d = Device::Cpu;
    let labels = Tensor::from_vec([0u8, 2u8, 1u8, 3u8].to_vec(), &[4], &d)?;
    println!("labels shape: {:?}", labels.shape());
    let r = one_hot(&labels, 4, candle_core::DType::U8)?;
    println!("{:?}", r.to_vec2::<u8>());
    anyhow::Ok(())
}

#[test]
fn magika_test() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let tensor = Tensor::ones(&[1536], candle_core::DType::U8, &device)?;
    println!("tensor shape: {:?}", tensor.shape());
    // tensor = tensor.unsqueeze(0)?;
    let double_tensor = Tensor::stack(&vec![tensor.copy()?, tensor], 0)?;
    let vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
    let model = Magika::new(vs.clone(), 113)?;
    let out = model.forward(&double_tensor)?;
    println!("out shape {:?}", out.shape());

    anyhow::Ok(())
}
