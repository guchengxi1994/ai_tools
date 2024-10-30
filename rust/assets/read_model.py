import onnx
import onnx.helper

model = onnx.load(r"model.onnx")

onnx.checker.check_model(model)
inferred_model = onnx.shape_inference.infer_shapes(model)
print(inferred_model.graph.input)  # 检查输入形状

# 遍历所有的值并转换 Int32 类型
for tensor in model.graph.initializer:
    print( f"tensor name {tensor.name}" )
    print( f"data_type {tensor.data_type}" )
    print( f"dims {tensor.dims}" )
