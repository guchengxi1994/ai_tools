use flutter_rust_bridge::frb;

use crate::{
    cv::{ClassificationResults, DetectResults, LOAD_MODEL_STATE_SINK},
    frb_generated::StreamSink,
    utils::image_processor::ImageProcessor,
};

#[deprecated]
pub fn classify_image(s: String, model_path: String, classes: Option<Vec<String>>) -> String {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let r = rt.block_on(async {
        let img = ImageProcessor::load_image(&s, 224, None)?;
        let model = candle_onnx::read_file(model_path)?;
        let graph = model.graph.as_ref().unwrap();
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(graph.input[0].name.to_string(), img.unsqueeze(0)?);
        let mut outputs = candle_onnx::simple_eval(&model, inputs)?;
        let output = outputs.remove(&graph.output[0].name).unwrap();

        let prs = candle_core::IndexOp::i(&output, 0)?.to_vec1::<f32>()?;

        let mut top: Vec<_> = prs.iter().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = top.into_iter().take(5).collect::<Vec<_>>();

        let cls;

        if let Some(c) = classes {
            cls = c;
        } else {
            cls = crate::constant::CLASSES
                .to_vec()
                .iter()
                .map(|&s| s.to_string())
                .collect();
        }

        anyhow::Ok(cls[top[0].0].clone())
    });

    match r {
        Ok(_r) => _r.to_owned(),
        Err(_e) => format!("error  {:?}", _e),
    }
}

#[deprecated]
pub fn yolov8_init(model_path: Option<String>) {
    let _ = crate::cv::yolov8::infer::init_yolov8_n(model_path);
}

#[deprecated]
pub fn yolov8_cleanup() {}

#[deprecated]
pub fn yolov8_detect(img: Vec<u8>) -> Vec<crate::cv::object_detect_result::ObjectDetectResult> {
    let r = crate::cv::yolov8::infer::yolov8n_detect(img);
    match r {
        Ok(_r) => _r,
        Err(_e) => {
            println!("[rust yolov8n] error  {:?}", _e);
            return vec![];
        }
    }
}

#[frb(sync)]
pub fn load_model_state_stream(s: StreamSink<String>) -> anyhow::Result<()> {
    let mut stream = LOAD_MODEL_STATE_SINK.write().unwrap();
    *stream = Some(s);
    anyhow::Ok(())
}

pub fn load_efficientnet(model_path: String) {
    let mut models = crate::cv::CV_MODELS.write().unwrap();
    let r = models.set_efficientnet(model_path);
    match r {
        Ok(_) => {
            if let Some(s) = LOAD_MODEL_STATE_SINK.read().unwrap().as_ref() {
                let _ = s.add("efficientnet => loaded".to_string());
            }
        }
        Err(e) => println!("[rust] error  {:?}", e),
    }
}

pub fn load_beit(model_path: String) {
    let mut models = crate::cv::CV_MODELS.write().unwrap();
    let r = models.set_beit(model_path);
    match r {
        Ok(_) => {
            if let Some(s) = LOAD_MODEL_STATE_SINK.read().unwrap().as_ref() {
                let _ = s.add("beit => loaded".to_string());
            }
        }
        Err(e) => println!("[rust] error  {:?}", e),
    }
}

pub fn load_yolov8(model_path: String) {
    let mut models = crate::cv::CV_MODELS.write().unwrap();
    let r = models.set_yolov8(model_path);
    match r {
        Ok(_) => {
            if let Some(s) = LOAD_MODEL_STATE_SINK.read().unwrap().as_ref() {
                let _ = s.add("yolov8 => loaded".to_string());
            }
        }
        Err(e) => println!("[rust] error  {:?}", e),
    }
}

pub fn run_classification(img: String, top_n: Option<usize>) -> ClassificationResults {
    let start = std::time::Instant::now();
    let model = crate::cv::CV_MODELS.read().unwrap();
    let r = model.run_classification(img, top_n.unwrap_or(5));
    let duration = start.elapsed().as_secs_f64();
    match r {
        Ok(r) => ClassificationResults {
            results: r,
            duration,
        },
        Err(e) => {
            println!("[rust] error  {:?}", e);
            return ClassificationResults {
                results: vec![],
                duration,
            };
        }
    }
}

pub fn run_detect(img: String) -> DetectResults {
    let start = std::time::Instant::now();
    let model = crate::cv::CV_MODELS.read().unwrap();
    let r = model.run_detect(img);
    let duration = start.elapsed().as_secs_f64();
    match r {
        Ok(r) => DetectResults {
            results: r,
            duration,
        },
        Err(e) => {
            println!("[rust] error  {:?}", e);
            return DetectResults {
                results: vec![],
                duration,
            };
        }
    }
}

pub fn run_detect_in_bytes(img: Vec<u8>) -> DetectResults {
    let start = std::time::Instant::now();
    let model = crate::cv::CV_MODELS.read().unwrap();
    let r = model.run_detect_in_bytes(img);
    let duration = start.elapsed().as_secs_f64();
    match r {
        Ok(r) => DetectResults {
            results: r,
            duration,
        },
        Err(e) => {
            println!("[rust] error  {:?}", e);
            return DetectResults {
                results: vec![],
                duration,
            };
        }
    }
}
