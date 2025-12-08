"""
Demo 1: 使用 GCM_model.h5 预测患者 2035_0_20210629 的血糖值
使用前9个数据点预测15/30/45/60分钟的血糖值
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Demo 1: 使用 GCM_model.h5 预测血糖值")
print("=" * 60)

# 1. 加载特征配置
print("\n[1] 加载特征配置...")
with open('pre-process/time_serise_attribute.json', 'r') as file:
    time_serise_attribute = json.load(file)
with open('pre-process/static_attribute.json', 'r') as file:
    static_attribute = json.load(file)

print(f"时序特征数量: {len(time_serise_attribute)}")
print(f"静态特征数量: {len(static_attribute)}")

# 2. 加载所有训练数据以创建scalers（与训练时一致）
print("\n[2] 加载所有训练数据以创建scalers...")
tmp_folder = 'pre-process/tmp_data'
tmp_files = os.listdir(tmp_folder)

all_data = []
for file in tmp_files:
    if file.endswith('.csv'):
        patient_data = pd.read_csv(os.path.join(tmp_folder, file))
        all_data.append(patient_data)

data = pd.concat(all_data, ignore_index=True)
data = data.drop(columns=['Date'])

target_attribute = ['15 min', '30 min', '45 min', '60 min']

# 分离特征和目标值
time_series_features = data[time_serise_attribute].values
static_features = data[static_attribute].values
targets = data[target_attribute].values

print(f"总数据量: {len(data)} 条记录")

# 3. 创建序列并训练scalers
print("\n[3] 创建序列并训练scalers...")

def create_sequences(features, targets, static_features, time_steps=10):
    ts_X, static_X, y = [], [], []
    for i in range(len(features) - time_steps):
        ts_X.append(features[i:i+time_steps])
        static_X.append(static_features[i])
        y.append(targets[i+time_steps])
    return np.array(ts_X), np.array(static_X), np.array(y)

ts_X, static_X, y = create_sequences(time_series_features, targets, static_features)

# 训练scalers
scaler_ts_X = StandardScaler()
scaler_static_X = StandardScaler()
scaler_y = StandardScaler()

scaler_ts_X.fit(ts_X.reshape(-1, ts_X.shape[-1]))
scaler_static_X.fit(static_X)
scaler_y.fit(y)

print("Scalers训练完成")

# 4. 加载患者2035的数据
print("\n[4] 加载患者 2035_0_20210629 的数据...")
patient_file = 'pre-process/tmp_data/2035_0_20210629.csv'
patient_data = pd.read_csv(patient_file)

print(f"患者数据总条数: {len(patient_data)}")

# 显示患者基本信息
print("\n" + "=" * 60)
print("患者基本信息")
print("=" * 60)
patient_info = patient_data.iloc[0]
gender_text = "女性" if patient_info["Gender (Female=1, Male=2)"] == 1 else "男性"
diabetes_type = "T1DM (1型糖尿病)" if patient_info["Type of Diabetes"] == 1 else "T2DM (2型糖尿病)"

print(f"患者ID: 2035_0_20210629")
print(f"性别: {gender_text}")
print(f"年龄: {patient_info['Age (years)']:.0f} 岁")
print(f"身高: {patient_info['Height (m)']:.2f} m")
print(f"体重: {patient_info['Weight (kg)']:.1f} kg")
print(f"BMI: {patient_info['BMI (kg/m2)']:.2f}")
print(f"糖尿病类型: {diabetes_type}")
print(f"糖尿病病程: {patient_info['Duration of Diabetes  (years)']:.0f} 年")
print(f"HbA1c: {patient_info['HbA1c (mmol/mol)']:.2f} mmol/mol")
print(f"空腹血糖: {patient_info['Fasting Plasma Glucose (mg/dl)']:.1f} mg/dL")
print("=" * 60)

# 提取前10个数据点（用于10步时间窗口）
patient_data_subset = patient_data.iloc[:10].copy()

# 提取时序特征和静态特征
patient_ts_features = patient_data_subset[time_serise_attribute].values  # (10, 51)
patient_static_features = patient_data_subset[static_attribute].values[0]  # (30,) 静态特征对所有时间步都一样

# 获取前9个时间点的血糖值（用于可视化）
cgm_values = patient_data_subset['CGM (mg / dl)'].values[:9]
dates = patient_data_subset['Date'].values[:9]

print(f"\n前9个时间点的血糖值 (CGM): {cgm_values}")
print(f"时间范围: {dates[0]} -> {dates[-1]}")

# 5. 准备模型输入
print("\n[5] 准备模型输入...")

# 准备输入数据
ts_X_input = patient_ts_features.reshape(1, 10, -1)  # (1, 10, 51)
static_X_input = patient_static_features.reshape(1, -1)  # (1, 30)

print(f"\n输入数据概览:")
print(f"  时序输入形状: {ts_X_input.shape}")
print(f"    - 批次大小: 1")
print(f"    - 时间步数: 10 (150分钟历史窗口)")
print(f"    - 时序特征数: {ts_X_input.shape[2]}")
print(f"  静态输入形状: {static_X_input.shape}")
print(f"    - 静态特征数: {static_X_input.shape[1]}")

print(f"\n时序特征包括:")
print(f"  - CGM血糖读数")
print(f"  - 饮食摄入 (0/1)")
print(f"  - CSII胰岛素剂量 (基础+餐时)")
print(f"  - 皮下注射胰岛素剂量 (多种类型)")
print(f"  - 静脉注射胰岛素剂量 (多种类型)")
print(f"  - 非胰岛素降糖药物")

print(f"\n静态特征包括:")
print(f"  - 患者人口学特征 (性别、年龄、身高、体重、BMI)")
print(f"  - 糖尿病相关信息 (类型、病程、并发症)")
print(f"  - 临床指标 (HbA1c、空腹血糖、C肽、胰岛素水平)")
print(f"  - 血脂指标 (总胆固醇、甘油三酯、HDL、LDL)")
print(f"  - 肾功能指标 (肌酐、eGFR、尿酸)")

# 标准化
ts_X_input = scaler_ts_X.transform(ts_X_input.reshape(-1, ts_X_input.shape[-1])).reshape(ts_X_input.shape)
static_X_input = scaler_static_X.transform(static_X_input)

print(f"\n数据已标准化 (StandardScaler)")
print(f"  时序数据标准化完成")
print(f"  静态数据标准化完成")

# 6. 加载模型
print("\n[6] 加载模型 GCM_model.h5...")
model = keras.models.load_model('GCM_model.h5')
print("模型加载成功")
print(f"模型输入: {[inp.shape for inp in model.inputs]}")
print(f"模型输出: {model.output.shape}")

# 7. 进行预测
print("\n[7] 进行预测...")
y_pred_scaled = model.predict([ts_X_input, static_X_input])
print(f"预测输出形状: {y_pred_scaled.shape}")

# 反标准化
y_pred = scaler_y.inverse_transform(y_pred_scaled)

print("\n预测结果:")
print(f"  15分钟后血糖: {y_pred[0][0]:.2f} mg/dL")
print(f"  30分钟后血糖: {y_pred[0][1]:.2f} mg/dL")
print(f"  45分钟后血糖: {y_pred[0][2]:.2f} mg/dL")
print(f"  60分钟后血糖: {y_pred[0][3]:.2f} mg/dL")

# 8. 可视化结果
print("\n[8] 绘制血糖预测图...")

# 创建时间轴
time_points = list(range(0, len(cgm_values) * 15, 15))  # 0, 15, 30, ..., 120分钟
prediction_times = [time_points[-1] + 15, time_points[-1] + 30,
                   time_points[-1] + 45, time_points[-1] + 60]  # 预测的时间点

# 绘图
plt.figure(figsize=(12, 6))

# 历史血糖值
plt.plot(time_points, cgm_values, 'bo-', linewidth=2, markersize=8, label='Historical CGM', zorder=3)

# 预测血糖值
plt.plot(prediction_times, y_pred[0], 'rs-', linewidth=2, markersize=10,
         label='Predicted (GCM_model.h5)', zorder=3)

# 连接线
plt.plot([time_points[-1], prediction_times[0]], [cgm_values[-1], y_pred[0][0]],
         'r--', alpha=0.5, linewidth=1)

# 添加数值标签
for i, (t, v) in enumerate(zip(time_points, cgm_values)):
    plt.text(t, v + 5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

for i, (t, v) in enumerate(zip(prediction_times, y_pred[0])):
    plt.text(t, v + 5, f'{v:.1f}', ha='center', va='bottom', fontsize=9, color='red')

# 添加正常血糖范围阴影
plt.axhspan(70, 180, alpha=0.1, color='green', label='Normal Range (70-180 mg/dL)')

plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Blood Glucose (mg/dL)', fontsize=12)
plt.title('Blood Glucose Prediction - Patient 2035_0_20210629\nModel: GCM_model.h5',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图片
output_file = 'demo1_prediction_GCM_model.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n图片已保存: {output_file}")

plt.show()

print("\n" + "=" * 60)
print("Demo 1 完成!")
print("=" * 60)
