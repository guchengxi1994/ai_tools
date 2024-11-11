// copied and modified from https://github.com/huggingface/candle/blob/main/candle-examples/examples/yolo-v8/main.rs

use std::{io::Cursor, sync::RwLock};

use crate::{
    constant::COCO_CLASSES,
    cv::{object_detect_result::ObjectDetectResult, LOAD_MODEL_STATE_SINK},
};

use super::model::{Multiples, YoloV8};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use image::DynamicImage;
use once_cell::sync::Lazy;

pub static YOLOV8N: Lazy<RwLock<Option<YoloV8>>> = Lazy::new(|| RwLock::new(None));

pub fn init_yolov8_n(model_path: Option<String>) -> anyhow::Result<()> {
    let mp = model_path
        .unwrap_or(r"D:\github_repo\ai_tools\rust\assets\yolov8n.safetensors".to_string());

    let d = Device::cuda_if_available(0)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[mp], DType::F32, &d)? };
    let v8 = YoloV8::load(vb, Multiples::n(), 80)?;

    let mut _v8 = YOLOV8N.write().unwrap();
    *_v8 = Some(v8);

    if let Some(s) = LOAD_MODEL_STATE_SINK.read().unwrap().as_ref() {
        let _ = s.add("yolov8_n => loaded".to_string());
    }
    Ok(())
}

pub fn yolov8n_detect(img: Vec<u8>) -> anyhow::Result<Vec<ObjectDetectResult>> {
    let model = YOLOV8N.read().unwrap();
    let d = Device::cuda_if_available(0)?;
    if let Some(v8) = model.as_ref() {
        let original_image = image::ImageReader::new(Cursor::new(img))
            .with_guessed_format()?
            .decode()?;
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

        let results = detect_result(
            &predictions,
            original_image,
            width,
            height,
            0.1,
            0.45,
            Some(d),
            None,
        )?;
        return Ok(results);
    }

    anyhow::bail!("model not loaded")
}

pub fn detect_result(
    pred: &Tensor,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    device: Option<Device>,
    class_names: Option<Vec<String>>,
) -> anyhow::Result<Vec<ObjectDetectResult>> {
    let cls;

    if let Some(c) = class_names {
        cls = c;
    } else {
        cls = crate::constant::COCO_CLASSES
            .to_vec()
            .iter()
            .map(|&s| s.to_string())
            .collect();
    }

    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;

    let d = device.unwrap_or(Device::cuda_if_available(0)?);
    let pred = pred.to_device(&d)?;
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
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
    let mut results = vec![];
    for (class_index, bbox_list) in bboxes.iter().enumerate() {
        for bbox in bbox_list {
            let xmin = (bbox.xmin * w_ratio) as i32;
            let ymin = (bbox.ymin * h_ratio) as i32;
            let dx = ((bbox.xmax - bbox.xmin) * w_ratio) as i32;
            let dy = ((bbox.ymax - bbox.ymin) * h_ratio) as i32;
            let class_name = cls[class_index].clone();
            results.push(ObjectDetectResult {
                class_id: class_index,
                class_name,
                confidence: bbox.confidence,
                xmin,
                ymin,
                width: dx,
                height: dy,
            });
        }
    }

    anyhow::Ok(results)
}

fn detect(
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
#[cfg(test)]
mod tests {
    use candle_core::{DType, Tensor};
    use candle_nn::{Module, VarBuilder};

    use crate::cv::yolov8::{
        infer::{detect, detect_result},
        model::Multiples,
    };

    #[test]
    fn test_yolo_v8() -> anyhow::Result<()> {
        let d = candle_core::Device::cuda_if_available(0)?;
        let model = r"D:\github_repo\ai_tools\rust\assets\yolov8m.safetensors".to_string();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &d)? };

        let v8 = super::YoloV8::load(vb, Multiples::m(), 80)?;

        let image_name =
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

        let results = detect_result(
            &predictions,
            original_image,
            width,
            height,
            0.1,
            0.45,
            None,
            None,
        )?;

        for r in results {
            println!("{:?}", r);
        }

        anyhow::Ok(())
    }
}
