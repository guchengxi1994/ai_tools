use candle_core::{Device, Tensor};
use candle_nn::Module;

pub struct Magika;

#[allow(unused_assignments)]
impl Magika {
    pub fn forword(x: Tensor, device: Device) -> anyhow::Result<Tensor> {
        let dense1 = candle_nn::Linear::new(
            Tensor::randn(0f32, 1.0, (213, 512), &device)?,
            Some(Tensor::randn(0f32, 1.0, (213,), &device)?),
        );

        let layer_norm1_gamma = Tensor::ones((1, 512), candle_core::DType::F32, &device)?;
        let layer_norm1_beta = Tensor::zeros((1, 512), candle_core::DType::F32, &device)?;

        let layer_norm = candle_nn::LayerNorm::new(layer_norm1_gamma, layer_norm1_beta, 1e-5);

        let _dense2 = candle_nn::Linear::new(
            Tensor::randn(0f32, 1.0, (64, 257), &device)?,
            Some(Tensor::randn(0f32, 1.0, (64,), &device)?),
        );

        let mut input = x.clone();
        input = dense1.forward(&input)?;
        input = input.relu()?;
        input = layer_norm.forward(&input)?;

        anyhow::bail!("a")
    }
}
