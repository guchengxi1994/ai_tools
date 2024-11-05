// copied and modified from https://github.com/huggingface/candle/blob/main/candle-examples/examples/yolo-v8/main.rs

use crate::constant::COCO_CLASSES;

use super::model::{Multiples, YoloV8, YoloV8Pose};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use image::DynamicImage;

pub fn detect(
    pred: &Tensor,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
    device: Option<Device>,
) -> anyhow::Result<DynamicImage> {
    let d = device.unwrap_or(Device::cuda_if_available(0)?);
    let pred = pred.to_device(&d)?;
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);

    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font)?;
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", COCO_CLASSES[class_index], b);
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let dx = (b.xmax - b.xmin) * w_ratio;
            let dy = (b.ymax - b.ymin) * h_ratio;
            if dx >= 0. && dy >= 0. {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                    image::Rgb([255, 0, 0]),
                );
            }
            if legend_size > 0 {
                imageproc::drawing::draw_filled_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, legend_size),
                    image::Rgb([170, 0, 0]),
                );
                let legend = format!(
                    "{}   {:.0}%",
                    COCO_CLASSES[class_index],
                    100. * b.confidence
                );
                imageproc::drawing::draw_text_mut(
                    &mut img,
                    image::Rgb([255, 255, 255]),
                    xmin,
                    ymin,
                    ab_glyph::PxScale {
                        x: legend_size as f32 - 1.,
                        y: legend_size as f32 - 1.,
                    },
                    &font,
                    &legend,
                )
            }
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

#[allow(unused_imports)]
mod tests {
    use candle_core::{DType, Tensor};
    use candle_nn::{Module, VarBuilder};

    use crate::cv::yolov8::{infer::detect, model::Multiples};

    #[test]
    fn test_yolo_v8() -> anyhow::Result<()> {
        let d = candle_core::Device::cuda_if_available(0)?;
        let model = r"D:\github_repo\ai_tools\rust\assets\yolov8m.safetensors".to_string();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model],
                DType::F32,
                &d,
            )?
        };

        let v8 = super::YoloV8::load(vb, Multiples::m(), 80)?;

        let mut image_name =
            std::path::PathBuf::from(r"D:\github_repo\ai_tools\rust\assets\test_image.png");
        let original_image = image::ImageReader::open(&image_name)?.decode()?;
        let (width, height) = {
            let w = original_image.width() as usize;
            let h = original_image.height() as usize;
            if w < h {
                let w = w * 640 / h;
                // Sizes have to be divisible by 32.
                (w / 32 * 32, 640)
            } else {
                let h = h * 640 / w;
                (640, h / 32 * 32)
            }
        };

        let image_t = {
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(data, (img.height() as usize, img.width() as usize, 3), &d)?
                .permute((2, 0, 1))?
        };
        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;

        let predictions = v8.forward(&image_t)?.squeeze(0)?;

        let image_t = detect(
            &predictions,
            original_image,
            width,
            height,
            0.1,
            0.45,
            15,
            None,
        )?;
        image_name.set_extension("pp.jpg");
        println!("writing {image_name:?}");
        image_t.save(image_name)?;

        anyhow::Ok(())
    }
}
