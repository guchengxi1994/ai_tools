// https://github.com/Zeleni9/pytorch-wgan/blob/master/models/dcgan.py

use candle_nn::{
    batch_norm, conv_transpose2d,
    ops::{leaky_relu, sigmoid},
    BatchNorm, BatchNormConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, ModuleT,
    VarBuilder,
};

pub struct Generator {
    // Z latent vector 100
    conv1: ConvTranspose2d,
    bn1: BatchNorm,
    // relu

    // State (1024x4x4)
    conv2: ConvTranspose2d,
    bn2: BatchNorm,
    // relu

    // State (512x8x8)
    conv3: ConvTranspose2d,
    bn3: BatchNorm,
    // relu

    // State (256x16x16)
    last: ConvTranspose2d,
    // tanh
    is_train: bool,

    out_channel_size: usize,
}

impl Generator {
    pub fn new(vb: VarBuilder, is_train: bool, out_channel_size: usize) -> anyhow::Result<Self> {
        let mut conv2d_config = ConvTranspose2dConfig::default();
        conv2d_config.padding = 0;
        conv2d_config.stride = 1;
        let batch_norm_config = BatchNormConfig::default();

        let conv1 = conv_transpose2d(100, 1024, 4, conv2d_config, vb.pp("conv1"))?;
        let bn1 = batch_norm(1024, batch_norm_config, vb.pp("bn1"))?;

        conv2d_config.padding = 1;
        conv2d_config.stride = 2;
        let conv2 = conv_transpose2d(1024, 512, 4, conv2d_config, vb.pp("conv2"))?;
        let bn2 = batch_norm(512, batch_norm_config, vb.pp("bn2"))?;

        let conv3 = conv_transpose2d(512, 256, 4, conv2d_config, vb.pp("conv3"))?;
        let bn3 = batch_norm(256, batch_norm_config, vb.pp("bn3"))?;

        let conv4 = conv_transpose2d(256, out_channel_size, 4, conv2d_config, vb.pp("conv4"))?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            last: conv4,
            is_train,
            out_channel_size,
        })
    }
}

impl Module for Generator {
    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        // println!("generator1 shape {:?}", xs.shape());
        let xs = self.conv1.forward(xs)?;
        let xs = self.bn1.forward_t(&xs, self.is_train)?;
        let xs = xs.relu()?;

        // println!("generator2 shape {:?}", xs.shape());
        let xs = self.conv2.forward(&xs)?;
        let xs = self.bn2.forward_t(&xs, self.is_train)?;
        let xs = xs.relu()?;

        // println!("generator3 shape {:?}", xs.shape());
        let xs = self.conv3.forward(&xs)?;
        let xs = self.bn3.forward_t(&xs, self.is_train)?;
        let xs = xs.relu()?;

        // println!("generator4 shape {:?}", xs.shape());
        let xs = self.last.forward(&xs)?;

        // println!("generator5 shape {:?}", xs.shape());
        xs.tanh()
    }
}

pub struct Discriminator {
    // Image (Cx32x32)
    conv1: candle_nn::Conv2d,
    // leaky_relu(0.2)

    // State (256x16x16)
    conv2: candle_nn::Conv2d,
    bn2: BatchNorm,
    // leaky_relu(0.2)

    // State (512x8x8)
    conv3: candle_nn::Conv2d,
    bn3: BatchNorm,
    // leaky_relu(0.2)
    out: candle_nn::Conv2d,
    // sigmoid
    input_channel_size: usize,
}

impl Discriminator {
    pub fn new(vb: VarBuilder, input_channel_size: usize) -> anyhow::Result<Self> {
        let mut conv2d_config = candle_nn::Conv2dConfig::default();
        conv2d_config.padding = 1;
        conv2d_config.stride = 2;
        let conv1 = candle_nn::conv2d(input_channel_size, 256, 4, conv2d_config, vb.pp("conv1"))?;

        let batch_norm_config = BatchNormConfig::default();
        let conv2 = candle_nn::conv2d(256, 512, 4, conv2d_config, vb.pp("conv2"))?;
        let bn2 = batch_norm(512, batch_norm_config, vb.pp("bn2"))?;

        let conv3 = candle_nn::conv2d(512, 1024, 4, conv2d_config, vb.pp("conv3"))?;
        let bn3 = batch_norm(1024, batch_norm_config, vb.pp("bn3"))?;

        conv2d_config.stride = 1;
        conv2d_config.padding = 0;
        let out = candle_nn::conv2d(1024, 1, 4, conv2d_config, vb.pp("out"))?;

        Ok(Self {
            conv1,
            conv2,
            bn2,
            conv3,
            bn3,
            out,
            input_channel_size,
        })
    }
}

impl Module for Discriminator {
    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        // println!("discriminator1 shape {:?}", xs.shape());
        let xs = self.conv1.forward(xs)?;
        let xs = leaky_relu(&xs, 0.2)?;

        // println!("discriminator2 shape {:?}", xs.shape());
        let xs = self.conv2.forward(&xs)?;
        let xs = self.bn2.forward_t(&xs, true)?;
        let xs = leaky_relu(&xs, 0.2)?;

        // println!("discriminator3 shape {:?}", xs.shape());
        let xs = self.conv3.forward(&xs)?;
        let xs = self.bn3.forward_t(&xs, true)?;
        let xs = leaky_relu(&xs, 0.2)?;

        // println!("discriminator4 shape {:?}", xs.shape());
        let xs = self.out.forward(&xs)?;

        // println!("discriminator5 shape {:?}", xs.shape());
        sigmoid(&xs)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Optimizer, VarMap};

    use crate::tests::utils::{
        files_to_tensor, read_100_files, read_10_files, save_tensor_as_image,
        save_tensor_as_rgb_image,
    };

    use super::*;

    #[test]
    fn test_train() -> anyhow::Result<()> {
        let d = Device::cuda_if_available(0)?;

        let real_labels = candle_core::Tensor::ones((10,), DType::F32, &d)?;

        let varmap = VarMap::new();
        let varmap_g = VarMap::new();
        let vs_g = VarBuilder::from_varmap(&varmap_g, DType::F32, &d);
        let vs_d = VarBuilder::from_varmap(&varmap, DType::F32, &d);
        let model = Generator::new(vs_g.clone(), true, 256)?;
        let dis = Discriminator::new(vs_d.clone(), 256)?;

        let param = candle_nn::ParamsAdamW::default();
        let mut adam_g = candle_nn::AdamW::new(varmap_g.all_vars(), param)?;

        let train = candle_core::Tensor::randn(0.0f32, 1.0f32, &[10, 100, 1, 1], &d)?;
        let fake_imgs = model.forward(&train)?;

        let out = dis.forward(&fake_imgs)?.flatten_all()?;
        println!("out shape: {:?}", out.shape());
        let loss = candle_nn::loss::binary_cross_entropy_with_logit(&out, &real_labels)?;
        println!("loss: {}", loss);
        loss.backward()?;
        adam_g.backward_step(&loss)?;

        anyhow::Ok(())
    }

    #[test]
    fn train() -> anyhow::Result<()> {
        let count = 100;
        let d = Device::cuda_if_available(0)?;
        let train_path = r"D:\workspace\yellow2anime\realFaces\";
        let train_files = read_100_files(train_path.to_string())?;
        let train_tensor = files_to_tensor(train_files, &d, 32, 32, 3)?;

        let varmap_d = VarMap::new();
        let varmap_g = VarMap::new();
        let vs_g = VarBuilder::from_varmap(&varmap_g, DType::F32, &d);
        let vs_d = VarBuilder::from_varmap(&varmap_d, DType::F32, &d);
        let model = Generator::new(vs_g.clone(), true, 3)?;
        let dis = Discriminator::new(vs_d.clone(), 3)?;

        let mut param = candle_nn::ParamsAdamW::default();
        param.lr = 0.0002;
        param.beta1 = 0.5;
        let mut adam_g = candle_nn::AdamW::new(varmap_g.all_vars(), param.clone())?;
        let mut adam_d = candle_nn::AdamW::new(varmap_d.all_vars(), param.clone())?;

        for epoch in 0..20 {
            let real_labels = candle_core::Tensor::ones((count,), DType::F32, &d)?;
            let fake_labels = candle_core::Tensor::zeros((count,), DType::F32, &d)?;

            // Train discriminator
            let train = candle_core::Tensor::randn(0.0f32, 1.0f32, &[count, 100, 1, 1], &d)?;
            let output = dis.forward(&train_tensor)?;
            let d_loss_real = candle_nn::loss::binary_cross_entropy_with_logit(
                &output.flatten_all()?,
                &real_labels,
            )?;
            // let real_score = output.clone();

            let fake_images = model.forward(&train)?;
            let output = dis.forward(&fake_images)?;
            let d_loss_fake = candle_nn::loss::binary_cross_entropy_with_logit(
                &output.flatten_all()?,
                &fake_labels,
            )?;
            // let fake_score = output.clone();
            let d_loss = (d_loss_real + d_loss_fake)?;
            d_loss.backward()?;
            adam_d.backward_step(&d_loss)?;

            let fake_images = model.forward(&train)?;
            let output = dis.forward(&fake_images)?;
            let g_loss = candle_nn::loss::binary_cross_entropy_with_logit(
                &output.flatten_all()?,
                &real_labels,
            )?;

            g_loss.backward()?;
            adam_g.backward_step(&g_loss)?;

            eprintln!(
                "epoch: {}, g_loss: {:?}  d_loss: {:?}",
                epoch,
                g_loss.to_vec0::<f32>()?,
                d_loss.to_vec0::<f32>()?
            );
            std::io::stdout().flush()?;
        }

        varmap_g.save("generator.bin")?;

        anyhow::Ok(())
    }

    #[test]
    fn test_generator() -> anyhow::Result<()> {
        let d = Device::cuda_if_available(0)?;
        let mut varmap_g = VarMap::new();
        varmap_g.load("generator.bin")?;
        let vs_g = VarBuilder::from_varmap(&varmap_g, DType::F32, &d);
        let g = Generator::new(vs_g, false, 3)?;

        let noise = Tensor::randn(0.0f32, 1.0f32, &[1, 100, 1, 1], &d)?;
        let output = g.forward(&noise)?;
        println!("output shape: {:?}", output.shape());

        save_tensor_as_rgb_image(&output, "test_generator.png", 32, 32)?;

        anyhow::Ok(())
    }
}
