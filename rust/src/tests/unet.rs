use std::io::{self, Write};

use candle_core::{DType, Device, Tensor};
use candle_dataset_loader::dataset::Dataset;
use candle_nn::{
    batch_norm, conv2d, conv_transpose2d, BatchNorm, BatchNormConfig, Conv2d, ConvTranspose2d,
    ConvTranspose2dConfig, Module, ModuleT, VarBuilder, VarMap,
};
use candle_nn::{loss, Optimizer};
use image::{DynamicImage, ImageReader};

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
    pub fn new(vb: VarBuilder, is_train: bool) -> anyhow::Result<Self> {
        let left1 = DoubleConv::new("left1", 3, 64, vb.pp("left1"), is_train)?;
        let left2 = DoubleConv::new("left2", 64, 128, vb.pp("left2"), is_train)?;
        let left3 = DoubleConv::new("left3", 128, 256, vb.pp("left3"), is_train)?;
        let left4 = DoubleConv::new("left4", 256, 512, vb.pp("left4"), is_train)?;

        let center = DoubleConv::new("center", 512, 1024, vb.pp("center"), is_train)?;

        let mut conv2d_config = ConvTranspose2dConfig::default();
        conv2d_config.stride = 2;
        let up1 = conv_transpose2d(1024, 512, 2, conv2d_config, vb.pp("up1"))?;

        let right1 = DoubleConv::new("right1", 1024, 512, vb.pp("right1"), is_train)?;

        let up2 = conv_transpose2d(512, 256, 2, conv2d_config, vb.pp("up2"))?;
        let right2 = DoubleConv::new("right2", 512, 256, vb.pp("right2"), is_train)?;

        let up3 = conv_transpose2d(256, 128, 2, conv2d_config, vb.pp("up3"))?;
        let right3 = DoubleConv::new("right3", 256, 128, vb.pp("right3"), is_train)?;

        let up4 = conv_transpose2d(128, 64, 2, conv2d_config, vb.pp("up4"))?;
        let right4 = DoubleConv::new("right4", 128, 64, vb.pp("right4"), is_train)?;

        let output = conv2d(
            64,
            1,
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

fn calculate_unet_output(input_size: usize) -> usize {
    // 定义下采样的层数和每层卷积的像素减少量
    let downsample_layers = 4;
    let conv_reduction = 2; // 每次卷积减少 2 像素

    // 初始大小
    let mut size = input_size;

    // 下采样阶段
    for _ in 0..downsample_layers {
        // 每层包含两次卷积，每次卷积减少 2 个像素
        size -= conv_reduction * 2;
        // 经过池化减半
        size /= 2;
    }

    // 底部层卷积，执行两次卷积
    size -= conv_reduction * 2;

    // 上采样阶段
    for _ in 0..downsample_layers {
        // 上采样（尺寸加倍）
        size *= 2;
        // 每层包含两次卷积，每次卷积减少 2 个像素
        size -= conv_reduction * 2;
    }

    // 返回最终输出层的尺寸
    size
}

pub struct DoubleConv {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    pub is_train: bool,
}

impl DoubleConv {
    pub fn new(
        prefix: &str,
        in_channel: usize,
        out_channel: usize,
        vb: VarBuilder,
        is_train: bool,
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
            is_train,
        })
    }
}

impl candle_nn::Module for DoubleConv {
    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let xs = self.conv1.forward(xs)?;
        let xs = self.bn1.forward_t(&xs, self.is_train)?;
        let xs = xs.relu()?;
        let xs = self.conv2.forward(&xs)?;
        let xs = self.bn2.forward_t(&xs, self.is_train)?;
        Ok(xs.relu()?)
    }
}

impl UNet {
    pub fn infer(img: String, save: bool, binary: bool) -> anyhow::Result<Tensor> {
        let d = Device::cuda_if_available(0)?;
        let img = ImageReader::open(img)?.decode()?;
        let mut t = img_to_tensor(img, 3, 256, 256, &d)?;
        t = t.unsqueeze(0)?;
        let mut varmap = VarMap::new();
        varmap.load("unet_model.bin")?;
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &d);
        let net = Self::new(vs.clone(), false)?;
        let r = net.forward(&t)?;
        println!("output shape: {:?}", r.shape());
        if save {
            Self::save_tensor_as_image(&r, "output.png", 68, 68, binary)?
        }
        anyhow::Ok(r)
    }

    fn save_tensor_as_image(
        tensor: &Tensor,
        path: &str,
        width: usize,
        height: usize,
        binary: bool,
    ) -> anyhow::Result<()> {
        // 将 Tensor 转换为 68x68 的 f32 向量
        let data = tensor.squeeze(0)?.squeeze(0)?;
        let data = data.flatten(0, 1)?.to_vec1::<f32>()?;

        // 创建一个 68x68 的灰度图像
        let mut img = image::GrayImage::new(width as u32, height as u32);

        // 填充图像数据，进行归一化并转换为 u8 类型
        for (i, pixel) in data.iter().enumerate() {
            let x = (i % width) as u32;
            let y = (i / height) as u32;
            let p;
            if binary {
                if pixel < &0.5 {
                    p = 0u8;
                } else {
                    p = 255u8;
                }
            } else {
                p = (pixel * 255.0) as u8;
            }

            // 假设像素值在 0.0 到 1.0 范围内，映射到 0 到 255
            let pixel_value = p;
            img.put_pixel(x, y, image::Luma([pixel_value]));
        }

        // 保存图像
        img.save(path)?;

        Ok(())
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

        // let cropped1 = center_crop(&left4_clone, (56, 56))?;
        let cropped1 = center_crop(&left4_clone, (xs.shape().dims()[2], xs.shape().dims()[3]))?;
        println!("cropped1 shape: {:?}", cropped1.shape());
        let xs = candle_core::Tensor::cat(&[cropped1, xs], 1)?;
        let xs = self.right1.forward(&xs)?;
        println!("right1 shape: {:?}", xs.shape());

        let xs = self.up2.forward(&xs)?;
        println!("up2 shape: {:?}", xs.shape());
        // let cropped2 = center_crop(&left3_clone, (104, 104))?;
        let cropped2 = center_crop(&left3_clone, (xs.shape().dims()[2], xs.shape().dims()[3]))?;
        let xs = candle_core::Tensor::cat(&[cropped2, xs], 1)?;
        let xs = self.right2.forward(&xs)?;
        println!("right2 shape: {:?}", xs.shape());

        let xs = self.up3.forward(&xs)?;
        // let cropped3 = center_crop(&left2_clone, (200, 200))?;
        let cropped3 = center_crop(&left2_clone, (xs.shape().dims()[2], xs.shape().dims()[3]))?;
        let xs = candle_core::Tensor::cat(&[cropped3, xs], 1)?;
        let xs = self.right3.forward(&xs)?;
        println!("right3 shape: {:?}", xs.shape());

        let xs = self.up4.forward(&xs)?;
        // let cropped4 = center_crop(&left1_clone, (392, 392))?;
        let cropped4 = center_crop(&left1_clone, (xs.shape().dims()[2], xs.shape().dims()[3]))?;
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

    let unet = UNet::new(vs, false)?;
    let input = candle_core::Tensor::randn(1f32, 0.5f32, &[1, 3, 572, 572], &dev)?;
    let output = unet.forward(&input)?;
    println!("output shape: {:?}", output.shape());

    Ok(())
}

fn read_100_files(folder_path: String) -> anyhow::Result<Vec<String>> {
    let mut files = Vec::new();
    let mut count = 0;
    for entry in std::fs::read_dir(folder_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            count += 1;
            if count > 100 {
                break;
            }
            files.push(path.display().to_string());
        }
    }
    files.sort();
    Ok(files)
}

fn files_to_tensor(
    files: Vec<String>,
    d: &Device,
    width: u32,
    height: u32,
    depth: usize,
) -> anyhow::Result<Tensor> {
    let mut tensors: Vec<Tensor> = Vec::new();
    for file in files {
        let img = ImageReader::open(file)?.decode()?;
        let t = img_to_tensor(img, depth, width, height, d)?;
        tensors.push(t);
    }

    Ok(Tensor::stack(&tensors, 0)?)
}

fn img_to_tensor(
    src: DynamicImage,
    depth: usize,
    width: u32,
    height: u32,
    d: &Device,
) -> anyhow::Result<Tensor> {
    if depth == 1 {
        let i = DynamicImage::ImageLuma8(src.into_luma8()).resize_exact(
            width,
            height,
            image::imageops::FilterType::Triangle,
        );
        let data = i.into_luma8().into_raw();
        let data = Tensor::from_vec(data, (width as usize, height as usize, depth), &d)?
            .permute((2, 0, 1))?;
        let r = (data.to_dtype(DType::F32)? / 255.)?;
        return Ok(r);
    } else if depth == 3 {
        let i = DynamicImage::ImageRgb8(src.to_rgb8()).resize_exact(
            width,
            height,
            image::imageops::FilterType::Triangle,
        );
        // 转换为张量并归一化
        let data = i.into_rgb8().into_raw();
        let data = Tensor::from_vec(data, (width as usize, height as usize, depth), &d)?
            .permute((2, 0, 1))?;
        let r = (data.to_dtype(DType::F32)? / 255.)?;
        return Ok(r);
    } else {
        panic!("Invalid depth value. Only 1 and 3 are supported.");
    }
}

#[test]
fn test_unet_size() {
    println!("{:?}", calculate_unet_output(572));
    println!("{:?}", calculate_unet_output(256));
}

#[test]
fn train_unet() -> anyhow::Result<()> {
    let d = Device::cuda_if_available(0)?;
    let train_path = r"D:\github_repo\ai_tools\rust\assets\isic2018\train\images\";
    let mask_path = r"D:\github_repo\ai_tools\rust\assets\isic2018\train\masks\";

    let train_files = read_100_files(train_path.to_string())?;
    let mask_files = read_100_files(mask_path.to_string())?;

    let train_tensor = files_to_tensor(train_files, &d, 256, 256, 3)?;
    let mask_tensor = files_to_tensor(mask_files, &d, 68, 68, 1)?;

    println!("train_tensor shape: {:?}", train_tensor.shape());
    println!("mask_tensor shape: {:?}", mask_tensor.shape());

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &d);
    let model = UNet::new(vs.clone(), true)?;
    let param = candle_nn::ParamsAdamW::default();
    let mut adam = candle_nn::AdamW::new(varmap.all_vars(), param)?;

    let d = Dataset {
        train_data: train_tensor,
        train_labels: mask_tensor,
        test_data: None,
        test_labels: None,
        batch_size: 3,
    };
    let mut i = 0;
    let _ = d.into_iter().take(300).for_each(|(x, y)| {
        i += 1;
        let results = model.forward(&x).unwrap();
        let loss = loss::mse(&results, &y).unwrap();
        println!("epoch: {}, loss: {}", i, loss);
        io::stdout().flush().unwrap();
        adam.backward_step(&loss).unwrap();
    });

    varmap.save("unet_model.bin")?;
    anyhow::Ok(())
}

#[test]
fn test_flatten() -> anyhow::Result<()> {
    // 创建一个二维张量，例如大小为 3x4
    let tensor = Tensor::new(
        &[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        &Device::Cpu,
    )?;

    // 将二维张量 flatten 到一维
    let flattened_tensor = tensor.flatten(0, 1)?;
    println!("{:?}", flattened_tensor.to_vec1::<f64>()?);
    Ok(())
}

#[test]
fn test_infer_unet() -> anyhow::Result<()> {
    UNet::infer(
        r"D:\github_repo\ai_tools\rust\assets\isic2018\train\images\0.png".to_owned(),
        true,
        false,
    )?;

    anyhow::Ok(())
}
