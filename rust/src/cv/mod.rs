use std::sync::RwLock;

use crate::frb_generated::StreamSink;

pub mod object_detect_result;
pub mod yolov8;

pub static LOAD_MODEL_STATE_SINK: RwLock<Option<StreamSink<String>>> = RwLock::new(None);
