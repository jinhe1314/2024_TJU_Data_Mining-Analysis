"""
Demo 3: 使用 glucose_predictor.tflite 预测患者 2035_0_20210629 的血糖值
使用前9个数据点预测15/30/45/60分钟的血糖值
"""

import tensorflow as tf
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
print("Demo 3: 使用 glucose_predictor.tflite 预测血糖值")
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

# 6. 加载TFLite模型
print("\n[6] 加载TFLite模型 glucose_predictor.tflite...")
tflite_model_path = 'mobile_deployment/mobile_deployment/src/models/glucose_predictor.tflite'

# 创建解释器
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 获取输入输出详情
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite模型加载成功")
print(f"输入详情:")
for i, detail in enumerate(input_details):
    print(f"  输入 {i}: 形状={detail['shape']}, 类型={detail['dtype']}")
print(f"输出详情:")
for i, detail in enumerate(output_details):
    print(f"  输出 {i}: 形状={detail['shape']}, 类型={detail['dtype']}")

# 7. 进行预测
print("\n[7] 进行预测...")

# 转换为float32（TFLite通常需要float32）
ts_X_input_float32 = ts_X_input.astype(np.float32)
static_X_input_float32 = static_X_input.astype(np.float32)

# 预测1: 使用完整输入（时序 + 患者静态特征）
print("\n  [7.1] 使用完整输入（时序特征 + 患者静态特征）...")
interpreter.set_tensor(input_details[0]['index'], ts_X_input_float32)
interpreter.set_tensor(input_details[1]['index'], static_X_input_float32)
interpreter.invoke()
y_pred_full_scaled = interpreter.get_tensor(output_details[0]['index'])
y_pred_full = scaler_y.inverse_transform(y_pred_full_scaled)

print("\n  完整输入预测结果:")
print(f"    15分钟后血糖: {y_pred_full[0][0]:.2f} mg/dL")
print(f"    30分钟后血糖: {y_pred_full[0][1]:.2f} mg/dL")
print(f"    45分钟后血糖: {y_pred_full[0][2]:.2f} mg/dL")
print(f"    60分钟后血糖: {y_pred_full[0][3]:.2f} mg/dL")

# 预测2: 仅使用时序特征（静态特征设为零向量）
print("\n  [7.2] 仅使用时序特征（不使用患者基本信息）...")
static_X_input_zero = np.zeros_like(static_X_input).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], ts_X_input_float32)
interpreter.set_tensor(input_details[1]['index'], static_X_input_zero)
interpreter.invoke()
y_pred_no_static_scaled = interpreter.get_tensor(output_details[0]['index'])
y_pred_no_static = scaler_y.inverse_transform(y_pred_no_static_scaled)

print("\n  仅时序特征预测结果:")
print(f"    15分钟后血糖: {y_pred_no_static[0][0]:.2f} mg/dL")
print(f"    30分钟后血糖: {y_pred_no_static[0][1]:.2f} mg/dL")
print(f"    45分钟后血糖: {y_pred_no_static[0][2]:.2f} mg/dL")
print(f"    60分钟后血糖: {y_pred_no_static[0][3]:.2f} mg/dL")

# 计算差异
print("\n  预测差异分析:")
for i, time in enumerate(['15分钟', '30分钟', '45分钟', '60分钟']):
    diff = y_pred_full[0][i] - y_pred_no_static[0][i]
    print(f"    {time}: {diff:+.2f} mg/dL (患者信息的影响)")

# 8. 可视化结果
print("\n[8] 绘制血糖预测图...")

# 创建时间轴
time_points = list(range(0, len(cgm_values) * 15, 15))  # 0, 15, 30, ..., 120分钟
prediction_times = [time_points[-1] + 15, time_points[-1] + 30,
                   time_points[-1] + 45, time_points[-1] + 60]  # 预测的时间点

# 绘图
fig, ax = plt.subplots(figsize=(14, 7))

# 历史血糖值
ax.plot(time_points, cgm_values, 'bo-', linewidth=2, markersize=8, label='Historical CGM', zorder=3)

# 预测线1: 使用完整输入（时序 + 患者信息）
ax.plot(prediction_times, y_pred_full[0], 'rs-', linewidth=2, markersize=10,
         label='With Patient Info (Time-series + Static features)', zorder=3)

# 预测线2: 仅使用时序特征（不使用患者信息）
ax.plot(prediction_times, y_pred_no_static[0], 'go-', linewidth=2, markersize=10,
         label='Without Patient Info (Time-series only)', zorder=3, alpha=0.7)

# 连接线
ax.plot([time_points[-1], prediction_times[0]], [cgm_values[-1], y_pred_full[0][0]],
         'r--', alpha=0.5, linewidth=1)
ax.plot([time_points[-1], prediction_times[0]], [cgm_values[-1], y_pred_no_static[0][0]],
         'g--', alpha=0.5, linewidth=1)

# 添加数值标签 - 历史值
for i, (t, v) in enumerate(zip(time_points, cgm_values)):
    ax.text(t, v + 5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

# 添加数值标签 - 完整输入预测
for i, (t, v) in enumerate(zip(prediction_times, y_pred_full[0])):
    ax.text(t, v + 5, f'{v:.1f}', ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

# 添加数值标签 - 仅时序预测
for i, (t, v) in enumerate(zip(prediction_times, y_pred_no_static[0])):
    ax.text(t, v - 8, f'{v:.1f}', ha='center', va='top', fontsize=8, color='green')

# 添加正常血糖范围阴影
ax.axhspan(70, 180, alpha=0.1, color='green', label='Normal Range (70-180 mg/dL)')

# 添加患者信息文本框
patient_info_text = f"""Patient Info:
ID: 2035_0_20210629
Gender: {gender_text} | Age: {patient_info['Age (years)']:.0f}y | BMI: {patient_info['BMI (kg/m2)']:.1f}
Type: {diabetes_type.split()[0]} | Duration: {patient_info['Duration of Diabetes  (years)']:.0f}y
HbA1c: {patient_info['HbA1c (mmol/mol)']:.1f} mmol/mol | FPG: {patient_info['Fasting Plasma Glucose (mg/dl)']:.1f} mg/dL

Input: 10 timesteps × 51 features + 30 static features
Model: TFLite (202KB, optimized for mobile)"""

# 在图表右上角添加文本框
ax.text(0.98, 0.97, patient_info_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
ax.set_title('Blood Glucose Prediction - Patient 2035_0_20210629\nModel: glucose_predictor.tflite',
          fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图片
output_file = 'demo3_prediction_tflite.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n图片已保存: {output_file}")

plt.show()

print("\n" + "=" * 60)
print("Demo 3 完成!")
print("=" * 60)
