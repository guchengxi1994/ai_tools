use candle_core::Tensor;

pub trait CsvLoad {
    fn load_csv(path: &str) -> anyhow::Result<(Tensor, Tensor)>;

    fn load_csv_without_header(path: &str) -> anyhow::Result<(Tensor, Tensor)>;
}
