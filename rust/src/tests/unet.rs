use candle_core::{DType, Tensor};
use candle_nn::{
    batch_norm, conv2d, conv_transpose2d, BatchNorm, BatchNormConfig, Conv2d, ConvTranspose2d,
    ConvTranspose2dConfig, Module, ModuleT, VarBuilder, VarMap,
};

pub struct UNet {
    left1: DoubleConv,
    left2: DoubleConv,
    left3: DoubleConv,
    left4: DoubleConv,
    center: DoubleConv,
    up1: ConvTranspose2d,
    right1: DoubleConv,
    up2: ConvTranspose2d,
    right2: DoubleConv,
    up3: ConvTranspose2d,
    right3: DoubleConv,
    up4: ConvTranspose2d,
    right4: DoubleConv,
    outc: Conv2d,
}

impl UNet {
    pub fn new(vb: VarBuilder) -> anyhow::Result<Self> {
        let left1 = DoubleConv::new("left1", 3, 64, vb.pp("left1"))?;
        let left2 = DoubleConv::new("left2", 64, 128, vb.pp("left2"))?;
        let left3 = DoubleConv::new("left3", 128, 256, vb.pp("left3"))?;
        let left4 = DoubleConv::new("left4", 256, 512, vb.pp("left4"))?;

        let center = DoubleConv::new("center", 512, 1024, vb.pp("center"))?;

        let mut conv2d_config = ConvTranspose2dConfig::default();
        conv2d_config.stride = 2;
        let up1 = conv_transpose2d(1024, 512, 2, conv2d_config, vb.pp("up1"))?;

        let right1 = DoubleConv::new("right1", 1024, 512, vb.pp("right1"))?;

        let up2 = conv_transpose2d(512, 256, 2, conv2d_config, vb.pp("up2"))?;
        let right2 = DoubleConv::new("right2", 512, 256, vb.pp("right2"))?;

        let up3 = conv_transpose2d(256, 128, 2, conv2d_config, vb.pp("up3"))?;
        let right3 = DoubleConv::new("right3", 256, 128, vb.pp("right3"))?;

        let up4 = conv_transpose2d(128, 64, 2, conv2d_config, vb.pp("up4"))?;
        let right4 = DoubleConv::new("right4", 128, 64, vb.pp("right4"))?;

        let output = conv2d(
            64,
            3,
            1,
            candle_nn::Conv2dConfig::default(),
            vb.pp("output"),
        )?;

        Ok(Self {
            left1,
            left2,
            left3,
            left4,
            center,
            up1,
            up2,
            up3,
            up4,
            right1,
            right2,
            right3,
            right4,
            outc: output,
        })
    }
}

pub struct DoubleConv {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl DoubleConv {
    pub fn new(
        prefix: &str,
        in_channel: usize,
        out_channel: usize,
        vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        let mut conv_cfg = candle_nn::Conv2dConfig::default();
        conv_cfg.padding = 1;
        conv_cfg.stride = 1;
        conv_cfg.dilation = 2;
        let conv1 = conv2d(
            in_channel,
            out_channel,
            3,
            conv_cfg,
            vb.clone().pp(format!("{} conv1", prefix)),
        )?;
        let bn1 = batch_norm(
            out_channel,
            BatchNormConfig::default(),
            vb.clone().pp(format!("{} bn1", prefix)),
        )?;
        let conv2 = conv2d(
            out_channel,
            out_channel,
            3,
            conv_cfg,
            vb.clone().pp(format!("{} conv2", prefix)),
        )?;
        let bn2 = batch_norm(
            out_channel,
            BatchNormConfig::default(),
            vb.clone().pp(format!("{} bn2", prefix)),
        )?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
        })
    }
}

impl candle_nn::Module for DoubleConv {
    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let xs = self.conv1.forward(xs)?;
        let xs = self.bn1.forward_t(&xs, true)?;
        let xs = xs.relu()?;
        let xs = self.conv2.forward(&xs)?;
        let xs = self.bn2.forward_t(&xs, true)?;
        Ok(xs.relu()?)
    }
}

impl candle_nn::Module for UNet {
    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let xs = self.left1.forward(xs)?;
        let left1_clone = xs.clone();
        let xs = xs.max_pool2d((2, 2))?;
        println!("left1 shape: {:?}", xs.shape());
        let xs = self.left2.forward(&xs)?;
        let left2_clone = xs.clone();
        let xs = xs.max_pool2d((2, 2))?;
        println!("left2 shape: {:?}", xs.shape());
        let xs = self.left3.forward(&xs)?;
        let left3_clone = xs.clone();
        let xs = xs.max_pool2d((2, 2))?;
        println!("left3 shape: {:?}", xs.shape());
        let xs = self.left4.forward(&xs)?;
        let left4_clone = xs.clone();
        let xs = xs.max_pool2d((2, 2))?;
        println!("left4 shape: {:?}", xs.shape());

        let xs = self.center.forward(&xs)?;
        println!("center shape: {:?}", xs.shape());

        let xs = self.up1.forward(&xs)?;
        println!("up1 shape: {:?}", xs.shape());

        let cropped1 = center_crop(&left4_clone, (56, 56))?;
        println!("cropped1 shape: {:?}", cropped1.shape());
        let xs = candle_core::Tensor::cat(&[cropped1, xs], 1)?;
        let xs = self.right1.forward(&xs)?;
        println!("right1 shape: {:?}", xs.shape());

        let xs = self.up2.forward(&xs)?;
        println!("up2 shape: {:?}", xs.shape());
        let cropped2 = center_crop(&left3_clone, (104, 104))?;
        let xs = candle_core::Tensor::cat(&[cropped2, xs], 1)?;
        let xs = self.right2.forward(&xs)?;
        println!("right2 shape: {:?}", xs.shape());

        let xs = self.up3.forward(&xs)?;
        let cropped3 = center_crop(&left2_clone, (200, 200))?;
        let xs = candle_core::Tensor::cat(&[cropped3, xs], 1)?;
        let xs = self.right3.forward(&xs)?;
        println!("right3 shape: {:?}", xs.shape());

        let xs = self.up4.forward(&xs)?;
        let cropped4 = center_crop(&left1_clone, (392, 392))?;
        let xs = candle_core::Tensor::cat(&[cropped4, xs], 1)?;
        let xs = self.right4.forward(&xs)?;
        println!("right4 shape: {:?}", xs.shape());

        let xs = self.outc.forward(&xs)?;
        Ok(xs)
    }
}

fn center_crop(
    tensor: &Tensor,
    target_shape: (usize, usize),
) -> Result<Tensor, candle_core::Error> {
    let (h, w) = (tensor.dim(2).unwrap(), tensor.dim(3).unwrap());
    // 检查目标形状是否超过当前张量尺寸
    if target_shape.0 > h || target_shape.1 > w {
        panic!("Target shape exceeds tensor dimensions.");
    }

    let crop_y = (h - target_shape.0) / 2;
    let crop_x = (w - target_shape.1) / 2;

    // 在高度维度上裁剪
    let cropped = tensor
        .narrow(2, crop_y, target_shape.0)?
        // 在宽度维度上裁剪
        .narrow(3, crop_x, target_shape.1)?;
    Ok(cropped)
}

#[test]
fn test_unet() -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let unet = UNet::new(vs)?;
    let input = candle_core::Tensor::randn(1f32, 0.5f32, &[1, 3, 572, 572], &dev)?;
    let output = unet.forward(&input)?;
    println!("output shape: {:?}", output.shape());

    Ok(())
}
