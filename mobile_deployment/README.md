# 血糖预测模型移动端部署指南

本项目提供了一个基于LSTM + Cross-Attention架构的血糖预测模型的完整移动端部署解决方案，支持预测未来15、30、45、60分钟的血糖水平。

## 📊 模型概况

- **模型架构**: LSTM + Cross-Attention混合神经网络
- **原始模型**: GCM_model.h5 (1.86 MB)
- **TFLite模型**: glucose_predictor.tflite (0.20 MB)
- **压缩率**: 89%
- **推理性能**: 平均1.4ms，吞吐量715K QPS
- **输入**: 时序数据[10,51] + 静态特征[30]
- **输出**: 血糖预测[4] (15/30/45/60分钟)

## 🚀 快速开始

### 1. 环境要求

#### Android
- Android API Level 23+ (Android 6.0+)
- Android Studio 4.2+
- Kotlin/Java 1.8+
- 设备存储空间: 至少1MB

#### iOS
- iOS 11.0+
- Xcode 12.0+
- Swift 5.0+
- 设备存储空间: 至少1MB

### 2. 模型文件

下载并集成以下文件：

```
mobile_deployment/
├── models/
│   └── glucose_predictor.tflite    # TensorFlow Lite模型文件
└── src/
    └── output/
        ├── deployment_info.json    # 部署信息
        └── performance_report.json # 性能报告
```

### 3. 快速集成

#### Android (Kotlin/Java)

1. 添加依赖到 `build.gradle`:
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.3'
}
```

2. 复制模型文件到 `app/src/main/assets/`

3. 使用示例代码：
```java
GlucosePredictor predictor = new GlucosePredictor();
predictor.initialize(getAssets());

// 准备数据
float[][] timeSeriesData = generateTimeSeriesData(); // [10][51]
float[] staticData = generateStaticData();           // [30]

// 执行预测
Map<Integer, Float> predictions = predictor.predictGlucose(timeSeriesData, staticData);
```

#### iOS (Swift)

1. 添加TensorFlow Lite依赖到 `Podfile`:
```ruby
pod 'TensorFlowLiteSwift'
```

2. 复制模型文件到项目Bundle

3. 使用示例代码：
```swift
let predictor = GlucosePredictor()
try predictor.initialize()

// 准备数据
let timeSeriesData = generateTimeSeriesData() // [[Float]]
let staticData = generateStaticData()         // [Float]

// 执行预测
let predictions = try predictor.predictGlucose(timeSeriesData: timeSeriesData, staticData: staticData)
```

## 📱 详细集成指南

### Android 集成

#### 1. 项目设置

**build.gradle (Module: app)**
```gradle
android {
    aaptOptions {
        noCompress "tflite"
    }
}

dependencies {
    // 核心TensorFlow Lite库
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'

    // 支持库（推荐）
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.3'

    // GPU加速（可选）
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
}
```

#### 2. 模型集成

- 将 `glucose_predictor.tflite` 复制到 `app/src/main/assets/`
- 确保 `build.gradle` 中包含 `aaptOptions { noCompress "tflite" }`

#### 3. 核心代码实现

参考 `examples/android/GlucosePredictor.java` 获取完整实现。

**主要方法：**
```java
// 初始化模型
predictor.initialize(assetManager);

// 执行预测
Map<Integer, Float> predictions = predictor.predictGlucose(timeSeriesData, staticData);

// 获取模型信息
String info = predictor.getModelInfo();

// 性能测试
Map<Integer, Float>[] batchResults = predictor.predictBatch(batchTimeSeriesData, batchStaticData);
```

### iOS 集成

#### 1. 项目设置

**Podfile**
```ruby
target 'YourApp' do
  use_frameworks!

  # TensorFlow Lite Swift库
  pod 'TensorFlowLiteSwift'

  # GPU委托（可选）
  pod 'TensorFlowLiteGpu'
end
```

#### 2. 模型集成

- 将 `glucose_predictor.tflite` 添加到Xcode项目
- 确保文件添加到Target Bundle Resources

#### 3. 核心代码实现

参考 `examples/ios/GlucosePredictor.swift` 获取完整实现。

**主要方法：**
```swift
// 初始化模型
try predictor.initialize()

// 执行预测
let predictions = try predictor.predictGlucose(timeSeriesData: timeSeriesData, staticData: staticData)

// 获取模型信息
let info = predictor.getModelInfo()

// 批量预测
let batchResults = try predictor.predictBatch(batchData: (timeSeries: batchTimeSeries, static: batchStatic))
```

## 📊 性能优化

### 1. 内存优化

- 模型大小仅0.20MB，适合内存受限设备
- 使用ByteBuffer避免数组拷贝
- 及时释放interpreter资源

### 2. 推理优化

- 多线程推理：Android使用4线程，iOS使用CPU核心数
- 预分配张量，避免重复分配
- GPU加速支持（可选）

### 3. 批处理优化

```java
// Android - 批量预测
Map<Integer, Float>[] results = predictor.predictBatch(batchTimeSeriesData, batchStaticData);

// iOS - 批量预测
let results = try predictor.predictBatch(batchData: (timeSeries: batchTimeSeries, static: batchStatic))
```

### 4. 性能基准测试

| 指标 | 数值 | 评估 |
|------|------|------|
| 模型大小 | 0.20 MB | ✅ 优秀 |
| 平均推理时间 | 1.40 ms | ✅ 优秀 |
| P95推理时间 | 1.48 ms | ✅ 优秀 |
| 吞吐量 | 715,739 QPS | ✅ 优秀 |
| 内存占用 | < 20 MB | ✅ 良好 |

## 🧪 测试与验证

### 1. 功能测试

#### Android
```java
// 参考 examples/android/GlucosePredictionActivity.java
// 提供完整的UI测试界面和性能测试功能
```

#### iOS
```swift
// 参考 examples/ios/GlucosePredictionViewController.swift
// 提供完整的UI测试界面和性能测试功能
```

### 2. 性能测试

使用内置的性能测试功能：

```bash
# 运行性能测试脚本
python mobile_deployment/mobile_deployment/src/simple_performance_test.py
```

### 3. 输入验证

确保输入数据格式正确：

- **时序数据**: [10][51] - 10个时间步，每步51个特征
- **静态数据**: [30] - 30个静态患者特征
- **数据类型**: Float32，范围[0,1]（标准化数据）

## 🔧 高级配置

### 1. 量化选项

模型已优化，支持以下量化策略：

- **默认优化**: 平衡精度和性能
- **动态量化**: 更小模型，轻微精度损失
- **INT8量化**: 最小模型，适合极端内存限制

### 2. 硬件加速

#### Android
```java
Interpreter.Options options = new Interpreter.Options();
options.setNumThreads(4);
// GPU加速
options.addDelegate(new GpuDelegate());
```

#### iOS
```swift
// CPU多线程
let options = Interpreter.Options()
options.threadCount = ProcessInfo.processInfo.processorCount

// GPU委托（可选）
let delegates = [MetalDelegate()]
interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: delegates)
```

### 3. 错误处理

完整错误处理机制：

```java
try {
    predictor.initialize(getAssets());
    Map<Integer, Float> predictions = predictor.predictGlucose(timeSeriesData, staticData);
} catch (IOException e) {
    // 模型加载错误
} catch (IllegalArgumentException e) {
    // 输入数据错误
} catch (Exception e) {
    // 其他错误
}
```

## 📈 监控与日志

### 1. 性能监控

```java
// 监控推理时间
long startTime = System.currentTimeMillis();
Map<Integer, Float> predictions = predictor.predictGlucose(timeSeriesData, staticData);
long inferenceTime = System.currentTimeMillis() - startTime;

// 记录性能指标
Log.d("GlucosePredictor", "Inference time: " + inferenceTime + "ms");
```

### 2. 内存监控

```java
// 监控内存使用
Runtime runtime = Runtime.getRuntime();
long usedMemory = runtime.totalMemory() - runtime.freeMemory();
Log.d("GlucosePredictor", "Memory usage: " + (usedMemory / 1024 / 1024) + "MB");
```

## 🚨 常见问题

### Q: 模型加载失败
**A:** 检查模型文件路径和权限：
- 确保TFLite文件在正确位置
- 检查文件完整性
- 验证文件读取权限

### Q: 推理结果异常
**A:** 验证输入数据：
- 确保数据维度正确 [10][51] 和 [30]
- 检查数据类型（Float32）
- 验证数据范围是否已标准化

### Q: 推理速度慢
**A:** 优化配置：
- 使用多线程推理
- 启用硬件加速（GPU）
- 检查设备性能和系统负载

### Q: 内存占用过高
**A:** 优化内存使用：
- 及时释放interpreter
- 使用ByteBuffer避免数组拷贝
- 监控内存泄漏

## 📚 相关文档

- [TensorFlow Lite Android指南](https://www.tensorflow.org/lite/guide/android)
- [TensorFlow Lite iOS指南](https://www.tensorflow.org/lite/guide/ios)
- [TensorFlow Lite性能优化](https://www.tensorflow.org/lite/performance)
- [模型量化指南](https://www.tensorflow.org/lite/performance/model_optimization)

## 🤝 贡献

欢迎提交问题和改进建议：
1. 性能优化技巧
2. 错误处理改进
3. 新平台支持
4. 文档完善

## 📄 许可证

本项目遵循原始项目的许可证条款。