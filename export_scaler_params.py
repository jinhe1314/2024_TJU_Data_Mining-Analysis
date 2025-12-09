"""
导出标准化器参数到JSON文件，供Android应用使用
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("导出Scaler参数到JSON")
print("=" * 60)

# 1. 加载特征配置
print("\n[1] 加载特征配置...")
with open('pre-process/time_serise_attribute.json', 'r') as file:
    time_serise_attribute = json.load(file)
with open('pre-process/static_attribute.json', 'r') as file:
    static_attribute = json.load(file)

print(f"时序特征数量: {len(time_serise_attribute)}")
print(f"静态特征数量: {len(static_attribute)}")

# 2. 加载所有训练数据
print("\n[2] 加载所有训练数据...")
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

# 4. 导出参数到JSON
print("\n[4] 导出参数到JSON...")

scaler_params = {
    "time_series": {
        "mean": scaler_ts_X.mean_.tolist(),
        "std": scaler_ts_X.scale_.tolist()
    },
    "static": {
        "mean": scaler_static_X.mean_.tolist(),
        "std": scaler_static_X.scale_.tolist()
    },
    "target": {
        "mean": scaler_y.mean_.tolist(),
        "std": scaler_y.scale_.tolist()
    }
}

output_file = 'scaler_params.json'
with open(output_file, 'w') as f:
    json.dump(scaler_params, f, indent=2)

print(f"✓ Scaler参数已导出到: {output_file}")
print(f"  - 时序特征: mean={scaler_params['time_series']['mean'][:3]}..., std={scaler_params['time_series']['std'][:3]}...")
print(f"  - 静态特征: mean={scaler_params['static']['mean'][:3]}..., std={scaler_params['static']['std'][:3]}...")
print(f"  - 目标值: mean={scaler_params['target']['mean']}, std={scaler_params['target']['std']}")
