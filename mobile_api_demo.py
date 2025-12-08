"""
Mobile API Demo: 使用 TFLite 模型预测血糖值 - 移动端友好版本
返回JSON格式的预测结果，适合移动端调用

用法:
    python mobile_api_demo.py --patient 2035_0_20210629 --output json
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os
import sys
from datetime import datetime, timedelta

class GlucosePredictorAPI:
    """血糖预测API - 移动端优化版本"""

    def __init__(self, model_path='mobile_deployment/mobile_deployment/src/models/glucose_predictor.tflite'):
        """初始化预测器"""
        self.model_path = model_path
        self.interpreter = None
        self.scaler_ts_X = None
        self.scaler_static_X = None
        self.scaler_y = None
        self.time_serise_attribute = None
        self.static_attribute = None

        # 加载配置和模型
        self._load_config()
        self._train_scalers()
        self._load_model()

    def _load_config(self):
        """加载特征配置"""
        with open('pre-process/time_serise_attribute.json', 'r') as file:
            self.time_serise_attribute = json.load(file)
        with open('pre-process/static_attribute.json', 'r') as file:
            self.static_attribute = json.load(file)

    def _train_scalers(self):
        """训练标准化器（使用训练数据）"""
        # 加载所有训练数据
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
        time_series_features = data[self.time_serise_attribute].values
        static_features = data[self.static_attribute].values
        targets = data[target_attribute].values

        # 创建序列
        def create_sequences(features, targets, static_features, time_steps=10):
            ts_X, static_X, y = [], [], []
            for i in range(len(features) - time_steps):
                ts_X.append(features[i:i+time_steps])
                static_X.append(static_features[i])
                y.append(targets[i+time_steps])
            return np.array(ts_X), np.array(static_X), np.array(y)

        ts_X, static_X, y = create_sequences(time_series_features, targets, static_features)

        # 训练scalers
        self.scaler_ts_X = StandardScaler()
        self.scaler_static_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.scaler_ts_X.fit(ts_X.reshape(-1, ts_X.shape[-1]))
        self.scaler_static_X.fit(static_X)
        self.scaler_y.fit(y)

    def _load_model(self):
        """加载TFLite模型"""
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def load_patient_data(self, patient_id):
        """加载患者数据"""
        patient_file = f'pre-process/tmp_data/{patient_id}.csv'
        if not os.path.exists(patient_file):
            raise FileNotFoundError(f"患者数据文件不存在: {patient_file}")

        patient_data = pd.read_csv(patient_file)
        return patient_data

    def prepare_input(self, patient_data, num_points=9):
        """准备模型输入数据（使用前N个数据点）"""
        # 提取前N+1个数据点（需要N+1个来创建N个时间步的序列）
        time_series_data = patient_data[self.time_serise_attribute].values[:num_points+1]
        static_data = patient_data[self.static_attribute].values[0]

        # 获取CGM值和时间
        cgm_values = patient_data['CGM (mg / dl)'].values[:num_points]
        timestamps = pd.to_datetime(patient_data['Date'].values[:num_points])

        # 创建时序输入 (1, 10, 51)
        ts_X_input = time_series_data.reshape(1, -1, len(self.time_serise_attribute))
        ts_X_input = self.scaler_ts_X.transform(ts_X_input.reshape(-1, ts_X_input.shape[-1])).reshape(ts_X_input.shape)

        # 创建静态输入 (1, 30)
        static_X_input = static_data.reshape(1, -1)
        static_X_input = self.scaler_static_X.transform(static_X_input)

        # 获取患者基本信息（确保转换为Python原生类型）
        patient_info = {
            'patient_id': str(patient_data.iloc[0].get('ID', 'Unknown')),
            'gender': '男性' if float(static_data[0]) == 1 else '女性',
            'age': int(float(static_data[1])),
            'height': float(static_data[2]),
            'weight': float(static_data[3]),
            'bmi': float(static_data[4]),
            'diabetes_type': 'T1DM' if float(static_data[5]) == 1 else 'T2DM',
            'diabetes_duration': int(float(static_data[6])),
            'hba1c': float(static_data[7]),
            'fasting_glucose': float(static_data[8])
        }

        return ts_X_input, static_X_input, cgm_values, timestamps, patient_info

    def predict(self, ts_X_input, static_X_input):
        """执行预测"""
        # 转换为float32
        ts_X_input_float32 = ts_X_input.astype(np.float32)
        static_X_input_float32 = static_X_input.astype(np.float32)

        # 设置输入
        self.interpreter.set_tensor(self.input_details[0]['index'], ts_X_input_float32)
        self.interpreter.set_tensor(self.input_details[1]['index'], static_X_input_float32)

        # 执行推理
        self.interpreter.invoke()

        # 获取输出
        y_pred_scaled = self.interpreter.get_tensor(self.output_details[0]['index'])
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        return y_pred[0]

    def predict_all_scenarios(self, patient_id, num_points=9):
        """预测所有场景（完整输入、无患者信息、普通进餐、高热量进餐）"""
        # 加载患者数据
        patient_data = self.load_patient_data(patient_id)

        # 准备输入
        ts_X_input, static_X_input, cgm_values, timestamps, patient_info = self.prepare_input(patient_data, num_points)

        # 场景1: 完整输入（时序 + 患者信息）
        pred_full = self.predict(ts_X_input, static_X_input)

        # 场景2: 仅时序特征（不使用患者信息）
        static_X_zero = np.zeros_like(static_X_input)
        pred_no_static = self.predict(ts_X_input, static_X_zero)

        # 场景3: 普通进餐场景（Dietary intake = 1）
        ts_X_meal = ts_X_input.copy()
        ts_X_meal_raw = self.scaler_ts_X.inverse_transform(ts_X_meal.reshape(-1, ts_X_meal.shape[-1])).reshape(ts_X_meal.shape)
        ts_X_meal_raw[0, -1, 1] = 1.0  # Dietary intake = 1
        ts_X_meal = self.scaler_ts_X.transform(ts_X_meal_raw.reshape(-1, ts_X_meal_raw.shape[-1])).reshape(ts_X_meal_raw.shape)
        pred_meal = self.predict(ts_X_meal, static_X_input)

        # 场景4: 高热量进餐场景（Dietary intake = 3）
        ts_X_high_meal = ts_X_input.copy()
        ts_X_high_meal_raw = self.scaler_ts_X.inverse_transform(ts_X_high_meal.reshape(-1, ts_X_high_meal.shape[-1])).reshape(ts_X_high_meal.shape)
        ts_X_high_meal_raw[0, -1, 1] = 3.0  # Dietary intake = 3
        ts_X_high_meal = self.scaler_ts_X.transform(ts_X_high_meal_raw.reshape(-1, ts_X_high_meal_raw.shape[-1])).reshape(ts_X_high_meal_raw.shape)
        pred_high_meal = self.predict(ts_X_high_meal, static_X_input)

        # 计算预测时间点
        last_timestamp = timestamps[-1]
        prediction_times = [last_timestamp + timedelta(minutes=i) for i in [15, 30, 45, 60]]

        # 构建结果
        result = {
            'patient_info': patient_info,
            'historical_data': {
                'timestamps': [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
                'glucose_values': [float(v) for v in cgm_values],
                'time_range': {
                    'start': timestamps[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')
                }
            },
            'predictions': {
                'timestamps': [t.strftime('%Y-%m-%d %H:%M:%S') for t in prediction_times],
                'time_horizons': ['15min', '30min', '45min', '60min'],
                'scenarios': {
                    'full_input': {
                        'description': 'With Patient Info (Time-series + Static features)',
                        'values': [float(v) for v in pred_full]
                    },
                    'no_patient_info': {
                        'description': 'Without Patient Info (Time-series only)',
                        'values': [float(v) for v in pred_no_static]
                    },
                    'normal_meal': {
                        'description': 'Normal Meal (Dietary intake=1)',
                        'values': [float(v) for v in pred_meal]
                    },
                    'high_calorie_meal': {
                        'description': 'High-Calorie Meal (Dietary intake=3)',
                        'values': [float(v) for v in pred_high_meal]
                    }
                }
            },
            'impact_analysis': {
                'patient_info_impact': [
                    float(pred_full[i] - pred_no_static[i]) for i in range(4)
                ],
                'normal_meal_impact': [
                    float(pred_meal[i] - pred_full[i]) for i in range(4)
                ],
                'high_calorie_meal_impact': [
                    float(pred_high_meal[i] - pred_full[i]) for i in range(4)
                ]
            },
            'model_info': {
                'model_type': 'TFLite',
                'model_file': self.model_path,
                'model_size': '202KB',
                'input_shape': {
                    'time_series': [int(x) for x in self.input_details[0]['shape']],
                    'static': [int(x) for x in self.input_details[1]['shape']]
                }
            },
            'metadata': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'api_version': '1.0.0'
            }
        }

        return result


def main():
    """主函数"""
    print("=" * 80)
    print("Mobile API Demo: TFLite血糖预测器")
    print("=" * 80)

    # 初始化API
    print("\n[1] 初始化预测器...")
    api = GlucosePredictorAPI()
    print("✓ 模型加载完成")

    # 预测
    print("\n[2] 执行预测...")
    patient_id = '2035_0_20210629'
    result = api.predict_all_scenarios(patient_id, num_points=9)
    print("✓ 预测完成")

    # 输出JSON结果
    print("\n[3] 预测结果:")
    print("=" * 80)
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(output_json)

    # 保存到文件
    output_file = 'mobile_api_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_json)
    print(f"\n✓ 结果已保存到: {output_file}")

    # 显示简要摘要
    print("\n" + "=" * 80)
    print("预测摘要:")
    print("=" * 80)
    print(f"患者ID: {result['patient_info']['patient_id']}")
    print(f"性别: {result['patient_info']['gender']} | 年龄: {result['patient_info']['age']}岁 | BMI: {result['patient_info']['bmi']:.1f}")
    print(f"糖尿病类型: {result['patient_info']['diabetes_type']} | 病程: {result['patient_info']['diabetes_duration']}年")
    print(f"\n预测时间范围: 15/30/45/60分钟")
    print(f"\n完整输入预测: {result['predictions']['scenarios']['full_input']['values']}")
    print(f"普通进餐预测: {result['predictions']['scenarios']['normal_meal']['values']}")
    print(f"高热量进餐预测: {result['predictions']['scenarios']['high_calorie_meal']['values']}")
    print("\n影响分析:")
    print(f"  普通进餐影响: {result['impact_analysis']['normal_meal_impact']}")
    print(f"  高热量进餐影响: {result['impact_analysis']['high_calorie_meal_impact']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
