"""
导出患者2035_0_20210629的真实输入数据，供Android应用使用
"""

import pandas as pd
import numpy as np
import json

print("=" * 60)
print("导出患者2035_0_20210629的数据")
print("=" * 60)

# 1. 加载特征配置
with open('pre-process/time_serise_attribute.json', 'r') as file:
    time_serise_attribute = json.load(file)
with open('pre-process/static_attribute.json', 'r') as file:
    static_attribute = json.load(file)

# 2. 加载患者数据
patient_file = 'pre-process/tmp_data/2035_0_20210629.csv'
patient_data = pd.read_csv(patient_file)

print(f"\n患者数据总条数: {len(patient_data)}")

# 3. 提取时序特征和静态特征
time_series_features = patient_data[time_serise_attribute].values
static_features = patient_data[static_attribute].values

# 4. 使用前10个时间步的数据（对应demo中的前9个数据点）
input_data = time_series_features[:10]  # shape: (10, 51)
static_data = static_features[0]  # shape: (30,)

# 5. 提取CGM值用于显示
cgm_values = input_data[:9, 0]  # 前9个时间步的CGM值

print(f"\n时序特征形状: {input_data.shape}")
print(f"静态特征形状: {static_data.shape}")
print(f"\nCGM值 (前9个): {cgm_values}")

# 6. 导出为JSON
output_data = {
    "patient_id": "2035_0_20210629",
    "historical_glucose": cgm_values.tolist(),
    "time_series_features": input_data.flatten().tolist(),  # 转为一维数组 (510,)
    "static_features": static_data.tolist()
}

output_file = 'patient_2035_data.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ 患者数据已导出到: {output_file}")
print(f"  - CGM值: {cgm_values[:3]}...")
print(f"  - 时序特征: {len(output_data['time_series_features'])} 个值")
print(f"  - 静态特征: {len(output_data['static_features'])} 个值")

# 7. 显示部分时序特征（前3个时间步的前10个特征）
print(f"\n前3个时间步的前10个特征:")
for t in range(3):
    print(f"  时间步{t}: {input_data[t, :10]}")
