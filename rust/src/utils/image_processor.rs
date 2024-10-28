use candle_core::{quantized::QuantizedType, Device, Tensor};
use image::{DynamicImage, ImageReader};

pub struct ImageProcessor;

impl ImageProcessor {
    pub fn load_image(path: &str, size: usize, device: Option<Device>) -> anyhow::Result<Tensor> {
        let img = ImageReader::open(path)?.decode()?.to_rgb8();
        let img = DynamicImage::ImageRgb8(img).resize_exact(
            size as u32,
            size as u32,
            image::imageops::FilterType::Triangle,
        );
        let data = img.into_rgb8().into_raw();
        let t;
        if device.is_none() {
            t = Tensor::from_vec(data, (size, size, 3), &Device::Cpu)?.permute((2, 0, 1))?;
        } else {
            t = Tensor::from_vec(data, (size, size, 3), &device.unwrap())?.permute((2, 0, 1))?;
        }

        let r = (t.to_dtype(candle_core::DType::F32)? / 255.)?;
        anyhow::Ok(r)
    }

    pub fn load_image_with_mean_and_std(
        path: &str,
        size: usize,
        device: Option<Device>,
        mean: Option<Vec<f32>>,
        std: Option<Vec<f32>>,
    ) -> anyhow::Result<Tensor> {
        let m;
        if mean.is_none() || mean.clone().unwrap().size() != 3 {
            m = crate::constant::IMAGENET_MEAN;
        } else {
            m = mean.unwrap().try_into().unwrap();
        }

        let s;
        if std.is_none() || std.clone().unwrap().size() != 3 {
            s = crate::constant::IMAGENET_STD;
        } else {
            s = std.unwrap().try_into().unwrap();
        }

        let d;
        if device.is_none() {
            d = Device::Cpu;
        } else {
            d = device.unwrap();
        }

        let mean = Tensor::new(&m, &d)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&s, &d)?.reshape((3, 1, 1))?;

        let r = ImageProcessor::load_image(path, size, Some(d))?;

        anyhow::Ok(r.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }
}
