use std::sync::RwLock;

use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;
use candle_transformers::models::beit::BeitVisionTransformer;
use image::DynamicImage;
use image::ImageReader;
use object_detect_result::ObjectDetectResult;
use once_cell::sync::Lazy;
use yolov8::model::YoloV8;

use crate::frb_generated::StreamSink;

pub mod beit;
pub mod efficientnet;
pub mod object_detect_result;
pub mod yolov8;

pub static LOAD_MODEL_STATE_SINK: RwLock<Option<StreamSink<String>>> = RwLock::new(None);

pub const IMAGENET_MEAN: [f32; 3] = [0.485f32, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229f32, 0.224, 0.225];

pub static CV_MODELS: Lazy<RwLock<CvModels>> = Lazy::new(|| RwLock::new(CvModels::default()));

pub struct CvModels {
    pub beit: Option<Box<dyn ModelRun<Vec<(String, f32)>> + Send + Sync>>,
    pub efficientnet: Option<Box<dyn ModelRun<Vec<(String, f32)>> + Send + Sync>>,
    pub yolov8: Option<Box<dyn ModelRun<Vec<ObjectDetectResult>> + Send + Sync>>,
}

impl CvModels {
    pub fn default() -> Self {
        Self {
            beit: None,
            efficientnet: None,
            yolov8: None,
        }
    }

    pub fn run_detect(&self, image_path: String) -> anyhow::Result<Vec<ObjectDetectResult>> {
        if self.yolov8.is_none() {
            anyhow::bail!("No model loaded");
        }
        let results = self.yolov8.as_ref().unwrap().run(image_path, 1)?;
        anyhow::Ok(results)
    }

    pub fn run_detect_in_bytes(
        &self,
        image_bytes: Vec<u8>,
    ) -> anyhow::Result<Vec<ObjectDetectResult>> {
        if self.yolov8.is_none() {
            anyhow::bail!("No model loaded");
        }
        let results = self.yolov8.as_ref().unwrap().run_in_bytes(image_bytes)?;
        anyhow::Ok(results)
    }

    pub fn run_classification(
        &self,
        image_path: String,
        top_n: usize,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        if self.beit.is_none() && self.efficientnet.is_none() {
            anyhow::bail!("No model loaded");
        }
        if let Some(model) = &self.beit {
            return model.run(image_path, top_n);
        }

        if let Some(model) = &self.efficientnet {
            return model.run(image_path, top_n);
        }

        anyhow::bail!("unimplemented")
    }

    pub fn set_beit(&mut self, model_path: String) -> anyhow::Result<()> {
        if self.beit.is_some() {
            return anyhow::Ok(());
        }
        self.efficientnet = None;

        let mut model: Model<BeitVisionTransformer> =
            Model::<BeitVisionTransformer>::new(model_path);
        model.load()?;
        self.beit = Some(Box::new(model));
        anyhow::Ok(())
    }

    pub fn set_efficientnet(&mut self, model_path: String) -> anyhow::Result<()> {
        if self.efficientnet.is_some() {
            return anyhow::Ok(());
        }
        self.beit = None;

        let mut model: Model<ModelProto> = Model::<ModelProto>::new(model_path);
        model.load()?;
        self.efficientnet = Some(Box::new(model));
        anyhow::Ok(())
    }

    pub fn set_yolov8(&mut self, model_path: String) -> anyhow::Result<()> {
        if self.yolov8.is_some() {
            return anyhow::Ok(());
        }

        let mut model: Model<YoloV8> = Model::<YoloV8>::new(model_path);
        model.load()?;
        self.yolov8 = Some(Box::new(model));
        anyhow::Ok(())
    }
}

pub fn load_image384_beit_norm(p: String, device: &Device) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(anyhow::Error::msg)?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (384, 384, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], device)?.reshape((3, 1, 1))?;
    anyhow::Ok(
        (data.to_dtype(candle_core::DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?,
    )
}

pub fn load_image224_efficientnet_norm(p: String, device: &Device) -> anyhow::Result<Tensor> {
    let img = ImageReader::open(p)?.decode()?.to_rgb8();
    let img =
        DynamicImage::ImageRgb8(img).resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let data = img.into_rgb8().into_raw();
    let data = Tensor::from_vec(data, (224, 224, 3), &device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&IMAGENET_MEAN, device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&IMAGENET_STD, device)?.reshape((3, 1, 1))?;
    anyhow::Ok(
        (data.to_dtype(candle_core::DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?,
    )
}

#[derive(Debug)]
pub struct ClassificationResults {
    pub results: Vec<(String, f32)>,
    pub duration: f64,
}

#[derive(Debug)]
pub struct DetectResults {
    pub results: Vec<ObjectDetectResult>,
    pub duration: f64,
}

pub enum CvTask {
    ObjectDetect,
    ImageClassification,
    Segmentation,
}

pub struct Model<T> {
    pub inner: Option<T>,
    pub model_path: String,
    pub device: Device,
}

pub trait ModelRun<S> {
    fn run(&self, image_path: String, top_n: usize) -> anyhow::Result<S>;

    fn run_in_bytes(&self, image_bytes: Vec<u8>) -> anyhow::Result<S>;

    fn load(&mut self) -> anyhow::Result<()>;

    fn get_task_type(&self) -> CvTask;
}
