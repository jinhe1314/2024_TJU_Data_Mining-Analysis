# 血糖预测项目 - 模型清单

## 模型统计概览

本项目包含 **3个不同的模型文件**（2个训练模型 + 1个移动端模型）

---

## 一、训练好的Keras模型

### 1. GCM_model.h5
- **文件大小**: 1.9MB
- **创建时间**: 2024年11月15日 16:48
- **说明**: 原始训练模型
- **路径**: `/home/gitlab-runner/2024_TJU_Data_Mining-Analysis/GCM_model.h5`
- **用途**: 基础版本的血糖预测模型

### 2. GCM_model_tf213_new.h5
- **文件大小**: 1.9MB
- **创建时间**: 2024年11月15日 22:05
- **说明**: TensorFlow 2.13版本重新训练的模型
- **路径**: `/home/gitlab-runner/2024_TJU_Data_Mining-Analysis/GCM_model_tf213_new.h5`
- **用途**: 针对TF 2.13优化的新版本

**对比说明**:
- 两个模型使用相同的网络架构（LSTM + 交叉注意力）
- 属于不同的训练过程或数据集配置
- 可用于性能对比实验

---

## 二、移动端TFLite模型

### 3. glucose_predictor.tflite
- **文件大小**: 202KB（约为原始大小的10%）
- **创建时间**: 2024年11月17日 09:10
- **说明**: 从Keras模型转换的TensorFlow Lite模型
- **存储位置**: 2个副本
  - `mobile_deployment/mobile_deployment/src/models/glucose_predictor.tflite`
  - `mobile_deployment/mobile_deployment/src/output/glucose_predictor.tflite`
- **用途**: iOS/Android移动端部署

**优化特性**:
- 模型量化压缩（从1.9MB压缩到202KB）
- 支持移动设备推理
- 兼容iOS和Android平台

---

## 三、模型架构说明

所有模型共享相同的神经网络架构：

```
时序编码器: LSTM (64→56→48→40→36→32)
           +
静态特征编码器: Dense (64→56→48→40→36→32)
           ↓
     双向交叉注意力
           ↓
      解码器 (64→16→4)
           ↓
输出: [15min, 30min, 45min, 60min] 血糖预测值
```

### 输入特征
- **时序特征**: 53个（包括CGM读数、药物剂量、饮食摄入等）
- **静态特征**: 31个（患者人口学特征和临床指标）
- **时间窗口**: 10个时间步（150分钟历史数据）

### 输出
- **预测时间点**: 4个（15分钟、30分钟、45分钟、60分钟后）
- **预测目标**: 血糖水平（mg/dL）

---

## 四、模型性能

根据项目评估结果：

| 指标 | 数值 |
|------|------|
| 整体MAE | 12.245 mg/dL |
| 整体RMSE | 17.990 mg/dL |
| R² 分数 | 0.8885 |

**各时间点MAE**:
- 15分钟: ~10.4 mg/dL
- 30分钟: ~11.8 mg/dL
- 45分钟: ~13.2 mg/dL
- 60分钟: ~14.5 mg/dL

---

## 五、模型使用指南

### 加载Keras模型
```python
from tensorflow import keras

# 加载原始模型
model = keras.models.load_model('GCM_model.h5')

# 或加载新版本
model = keras.models.load_model('GCM_model_tf213_new.h5')
```

### 加载TFLite模型
```python
import tensorflow as tf

# 加载解释器
interpreter = tf.lite.Interpreter(
    model_path='mobile_deployment/mobile_deployment/src/models/glucose_predictor.tflite'
)
interpreter.allocate_tensors()

# 获取输入输出详情
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

---

## 六、模型文件路径清单

```
/home/gitlab-runner/2024_TJU_Data_Mining-Analysis/
├── GCM_model.h5                                   # Keras原始模型
├── GCM_model_tf213_new.h5                         # Keras新版本模型
└── mobile_deployment/
    └── mobile_deployment/
        └── src/
            ├── models/
            │   └── glucose_predictor.tflite       # TFLite模型（副本1）
            └── output/
                └── glucose_predictor.tflite       # TFLite模型（副本2）
```

---

**最后更新**: 2024年12月8日
