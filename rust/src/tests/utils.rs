use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, ImageReader, Rgb, RgbImage};

pub fn read_100_files(folder_path: String) -> anyhow::Result<Vec<String>> {
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

pub fn read_10_files(folder_path: String) -> anyhow::Result<Vec<String>> {
    let mut files = Vec::new();
    let mut count = 0;
    for entry in std::fs::read_dir(folder_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            count += 1;
            if count > 10 {
                break;
            }
            files.push(path.display().to_string());
        }
    }
    files.sort();
    Ok(files)
}

pub fn files_to_tensor(
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

pub fn img_to_tensor(
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

pub fn save_tensor_as_image(
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

pub fn save_tensor_as_rgb_image(
    tensor: &Tensor,
    path: &str,
    width: usize,
    height: usize,
) -> anyhow::Result<()> {
    // 将 Tensor 转换为 3*widthxheight 的 f32 向量
    let data = tensor.squeeze(0)?;
    let data = data.flatten(0, 2)?.to_vec1::<f32>()?;

    let mut img = RgbImage::new(width as u32, height as u32);

    for (i, chunk) in data.chunks(3).enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;

        let r = (chunk[0] * 255.0) as u8;
        let g = (chunk[1] * 255.0) as u8;
        let b = (chunk[2] * 255.0) as u8;
        println!("{},{},{}", r, g, b);
        // 归一化 RGB 数据并创建图像像素
        img.put_pixel(x, y, Rgb([r, g, b]));
    }

    // 保存图像
    img.save(path)?;

    Ok(())
}
