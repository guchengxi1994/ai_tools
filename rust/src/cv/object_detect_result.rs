#[derive(Debug)]
pub struct ObjectDetectResult {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub xmin: i32,
    pub ymin: i32,
    pub width: i32,
    pub height: i32,
}
