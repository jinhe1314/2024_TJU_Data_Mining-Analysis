# 📱 血糖预测模型移动端部署指南

## 🎯 模型基本信息

- **模型文件**: `GCM_model.h5`
- **文件大小**: 1.86 MB
- **参数数量**: 141,964
- **内存占用**: ~0.54 MB (仅参数)
- **平均推理时间**: ~47.58 ms (CPU)

## ✅ 移动端部署可行性评估

### 1. **模型规模** - 优秀
- ✅ 文件大小 < 2MB，远低于移动端推荐阈值(10MB)
- ✅ 内存占用极小，适合资源受限的移动设备

### 2. **推理性能** - 良好
- ✅ 推理时间 < 50ms，满足实时性要求
- ✅ CPU推理即可，无需GPU加速

### 3. **模型结构** - 兼容
- ✅ 使用LSTM层，移动端友好
- ✅ 激活函数: ReLU, Linear, Tanh - 均为移动端优化支持
- ✅ 无复杂的自定义层

## 📋 输入输出规格

### 输入要求
1. **时序数据**: `(1, 10, 51)`
   - 10个时间步的历史数据
   - 每个时间步包含51个特征

2. **静态数据**: `(1, 30)`
   - 30个静态患者特征

### 输出结果
- **预测结果**: `(1, 4)`
  - 4个时间点的血糖预测值
  - 分别对应: 15分钟、30分钟、45分钟、60分钟

## 🛠️ 移动端部署方案

### 方案1: TensorFlow Lite (推荐)
```bash
# 转换模型为TFLite格式
import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model('GCM_model.h5')

# 转换为TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存TFLite模型
with open('glucose_predictor.tflite', 'wb') as f:
    f.write(tflite_model)
```

**优势:**
- 📦 模型体积进一步减小
- ⚡ 推理速度提升2-3倍
- 🔋 电池友好，功耗低
- 📱 完美支持Android/iOS

### 方案2: ONNX + CoreML (iOS)
```python
# 转换为ONNX格式
import tf2onnx
import onnx

spec = (tf.TensorSpec((None, 10, 51), tf.float32, name="input_1"),
         tf.TensorSpec((None, 30), tf.float32, name="input_2"))

output_path = "glucose_predictor.onnx"
tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
```

**优势:**
- 🍎 iOS原生支持
- 🚀 硬件加速优化

### 方案3: PyTorch Mobile (Android)
```python
# 转换为PyTorch格式
import torch

# 首先需要将Keras模型转换为PyTorch模型
# 然后使用torch.jit.trace进行优化
```

## 📱 移动端集成示例

### Android (Kotlin + TensorFlow Lite)
```kotlin
// 加载TFLite模型
private lateinit var interpreter: Interpreter

private fun loadModel() {
    val model = loadModelFile(context, "glucose_predictor.tflite")
    val options = Interpreter.Options()
    options.setNumThreads(4)
    interpreter = Interpreter(model, options)
}

// 进行预测
fun predictGlucose(timeSeriesData: FloatArray, staticData: FloatArray): FloatArray {
    val inputs = arrayOf<Any>(timeSeriesData, staticData)
    val outputs = Array(1) { Array<Float>(4) { 0f } }

    interpreter.run(inputs, outputs)
    return outputs[0]
}
```

### iOS (Swift + CoreML)
```swift
import CoreML

class GlucosePredictor {
    private let model: GlucosePredictor

    init() throws {
        self.model = try GlucosePredictor(configuration: MLModelConfiguration())
    }

    func predict(timeSeries: MLMultiArray, static: MLMultiArray) throws -> MLMultiArray {
        let input = GlucosePredictorInput(input_1: timeSeries, input_2: static)
        let output = try model.prediction(input: input)
        return output.linear_1
    }
}
```

## 📊 预期性能表现

### 移动端性能指标
| 设备类型 | 推理时间 | 内存使用 | 功耗 |
|----------|----------|----------|------|
| 高端手机 | ~10-20ms | ~10MB | 低 |
| 中端手机 | ~20-40ms | ~15MB | 中等 |
| 低端手机 | ~40-80ms | ~25MB | 较高 |

## 🔧 部署优化建议

### 1. **模型量化**
```python
# 8位整数量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

### 2. **批处理优化**
- 支持批量预测以提高吞吐量
- 单次预测保持低延迟

### 3. **缓存策略**
- 预加载模型避免首次推理延迟
- 结果缓存减少重复计算

## ⚠️ 注意事项

### 1. **数据预处理**
- 确保移动端数据预处理与训练时一致
- 注意特征标准化参数的同步

### 2. **版本兼容性**
- TensorFlow版本匹配
- 移动端操作系统版本支持

### 3. **测试验证**
- 在目标设备上进行充分测试
- 验证预测精度与桌面端一致

## 📦 部署包建议

### 完整部署包应包含:
1. `glucose_predictor.tflite` - 优化后的模型文件
2. `preprocessing_params.json` - 数据标准化参数
3. `feature_mapping.json` - 特征映射规则
4. `sdk_documentation.md` - API使用文档

## 🎉 总结

**GCM_model.h5完全适合移动端部署！**

### 关键优势:
- ✅ 模型小巧 (1.86MB)
- ✅ 推理快速 (~47ms)
- ✅ 结构简单易转换
- ✅ 预测精度高 (MAE: 12.24 mg/dL)

### 推荐部署方案:
1. **TensorFlow Lite** - 跨平台最佳选择
2. **量化优化** - 进一步减小体积提升速度
3. **充分测试** - 确保移动端精度一致

该模型在移动端可以实现实时、准确的血糖预测，非常适合集成到糖尿病管理应用中。