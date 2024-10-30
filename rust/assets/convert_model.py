import onnx
from onnx import helper
from onnx import TensorProto

def convert_to_float(model_path, output_path):
    # 加载模型
    model = onnx.load(model_path)
    graph = model.graph

    # 遍历节点并检查数据类型
    for tensor in graph.initializer:
        # 检查是否为非 float 数据类型
        if tensor.data_type != TensorProto.FLOAT:
            # 转换数据类型为 float
            tensor.data_type = TensorProto.FLOAT
            # 需要重新设置数据，以匹配 float 类型
            tensor.float_data[:] = [float(i) for i in tensor.int32_data]
            tensor.int32_data[:] = []  # 清空原始数据

    # 保存转换后的模型
    onnx.save(model, output_path)
    print(f"模型已保存至 {output_path}")

# 使用示例
convert_to_float(r"D:\github_repo\ai_tools\rust\assets\model.onnx", "converted_model.onnx")