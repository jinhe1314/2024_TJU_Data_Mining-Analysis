#!/usr/bin/env python3
"""
血糖预测模型训练 - GPU优化简化版本
Blood Glucose Prediction Model Training - GPU Optimized Simplified
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import json
import time

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# 尝试设置GPU内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} device(s)")
    except:
        print("GPU memory growth setting failed")

class FastTimeModel:
    """
    GPU优化的时间序列模型
    """

    def __init__(self, model_path="GCM_model_fast.h5"):
        self.scaler_ts_X = StandardScaler()
        self.scaler_static_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model_save_path = model_path

        # 加载配置和数据
        self._load_attributes()
        self._load_data()
        self._create_sequences()
        self._standardize_data()
        self._split_data()

        self.model = None

    def _load_attributes(self):
        """加载特征属性配置"""
        with open('pre-process/time_serise_attribute.json', 'r') as file:
            self.time_series_attribute = json.load(file)
        with open('pre-process/static_attribute.json', 'r') as file:
            self.static_attribute = json.load(file)
        print(f"Time series features: {len(self.time_series_attribute)}")
        print(f"Static features: {len(self.static_attribute)}")

    def _load_data(self):
        """加载处理过的数据"""
        tmp_folder = 'pre-process/tmp_data'
        tmp_files = [f for f in os.listdir(tmp_folder) if f.endswith('.csv')]

        print(f"Loading {len(tmp_files)} patient files...")
        all_data = []

        for i, file in enumerate(tmp_files):
            if i % 20 == 0:
                print(f"Loading file {i+1}/{len(tmp_files)}: {file}")

            patient_data = pd.read_csv(os.path.join(tmp_folder, file))
            all_data.append(patient_data)

        self.data = pd.concat(all_data, ignore_index=True)
        self.data = self.data.drop(columns=['Date'], errors='ignore')

        print(f"Total data shape: {self.data.shape}")

    def _create_sequences(self):
        """创建时间序列序列"""
        target_attribute = ['15 min', '30 min', '45 min', '60 min']

        # 分离特征和目标值
        time_series_features = self.data[self.time_series_attribute].values
        static_features = self.data[self.static_attribute].values
        targets = self.data[target_attribute].values

        def create_sequences(features, targets, static_features, time_steps=10):
            ts_X, static_X, y = [], [], []
            for i in range(len(features) - time_steps):
                ts_X.append(features[i:i+time_steps])
                static_X.append(static_features[i])
                y.append(targets[i+time_steps])
            return np.array(ts_X), np.array(static_X), np.array(y)

        self.ts_X, self.static_X, self.y = create_sequences(
            time_series_features, targets, static_features
        )

        print(f"Sequences created - Time series: {self.ts_X.shape}, "
              f"Static: {self.static_X.shape}, Targets: {self.y.shape}")

    def _standardize_data(self):
        """数据标准化"""
        print("Standardizing data...")

        # 重塑并标准化时间序列数据
        original_shape = self.ts_X.shape
        self.ts_X = self.scaler_ts_X.fit_transform(
            self.ts_X.reshape(-1, self.ts_X.shape[-1])
        ).reshape(original_shape)

        # 标准化静态特征
        self.static_X = self.scaler_static_X.fit_transform(self.static_X)

        # 标准化目标值
        self.y = self.scaler_y.fit_transform(self.y)

        print("Data standardization completed")

    def _split_data(self):
        """分割训练测试数据"""
        self.ts_X_train, self.ts_X_test, self.static_X_train, self.static_X_test,
        self.y_train, self.y_test = train_test_split(
            self.ts_X, self.static_X, self.y,
            test_size=0.2, random_state=42
        )

        print(f"Train/Test split - Train: {self.ts_X_train.shape[0]}, "
              f"Test: {self.ts_X_test.shape[0]}")

    def build_model(self):
        """
        构建优化的混合神经网络模型
        """
        print("Building fast model...")

        # 时序数据输入
        ts_input = Input(shape=(self.ts_X_train.shape[1], self.ts_X_train.shape[2]),
                        name='timeseries_input')

        # 更高效的LSTM架构
        x = LSTM(128, return_sequences=True, name='lstm_1')(ts_input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = LSTM(64, return_sequences=False, name='lstm_2')(x)
        x = BatchNormalization()(x)
        ts_embedding = x

        # 静态特征输入
        static_input = Input(shape=(self.static_X_train.shape[1],),
                           name='static_input')

        y = Dense(64, activation='relu', name='dense_static_1')(static_input)
        y = BatchNormalization()(y)
        y = Dropout(0.3)(y)

        y = Dense(32, activation='relu', name='dense_static_2')(y)
        static_embedding = y

        # Cross-Attention机制 (简化版)
        # 使用点积注意力
        def attention(q, k):
            scores = tf.matmul(q, k, transpose_b=True)
            weights = tf.nn.softmax(scores, axis=-1)
            return tf.matmul(weights, k)

        # 计算注意力
        ts_q = tf.expand_dims(ts_embedding, axis=1)  # (batch, 1, features)
        static_k = tf.expand_dims(static_embedding, axis=1)  # (batch, 1, features)
        attended = attention(ts_q, static_k)
        attended = tf.squeeze(attended, axis=1)  # (batch, features)

        # 合并特征
        merged = Concatenate()([ts_embedding, attended, static_embedding])

        # 解码器
        z = Dense(128, activation='relu', name='decoder_1')(merged)
        z = BatchNormalization()(z)
        z = Dropout(0.3)(z)

        z = Dense(64, activation='relu', name='decoder_2')(z)
        z = BatchNormalization()(z)

        z = Dense(32, activation='relu', name='decoder_3')(z)
        z = Dropout(0.3)(z)

        output = Dense(4, name='output')(z)  # 4个目标值

        # 构建模型
        self.model = Model(
            inputs=[ts_input, static_input],
            outputs=output,
            name='BloodGlucosePredictionModel_Fast'
        )

        # 编译模型
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        return self.model

    def train_model(self, epochs=80, batch_size=64, validation_split=0.2):
        """
        训练模型
        """
        print(f"Training fast model for {epochs} epochs with batch size {batch_size}...")

        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # 开始训练
        start_time = time.time()

        self.history = self.model.fit(
            [self.ts_X_train, self.static_X_train],
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self.history.history

    def evaluate_model(self):
        """评估模型性能"""
        print("Evaluating model...")

        test_loss, test_mae = self.model.evaluate(
            [self.ts_X_test, self.static_X_test],
            self.y_test,
            batch_size=64,
            verbose=0
        )

        # 预测
        y_pred = self.model.predict([self.ts_X_test, self.static_X_test],
                                    batch_size=64)

        # 转换回原始尺度
        y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        y_test_orig = self.scaler_y.inverse_transform(self.y_test)

        # 计算每个时间点的MAE
        mae_per_horizon = np.mean(np.abs(y_pred_orig - y_test_orig), axis=0)

        results = {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'mae_15min': mae_per_horizon[0],
            'mae_30min': mae_per_horizon[1],
            'mae_45min': mae_per_horizon[2],
            'mae_60min': mae_per_horizon[3]
        }

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE (normalized): {test_mae:.4f}")
        print(f"MAE per horizon - 15min: {mae_per_horizon[0]:.2f}, "
              f"30min: {mae_per_horizon[1]:.2f}, "
              f"45min: {mae_per_horizon[2]:.2f}, "
              f"60min: {mae_per_horizon[3]:.2f}")

        return results

def main():
    """主函数"""
    print("=" * 60)
    print("血糖预测模型训练 - GPU优化版本")
    print("Blood Glucose Prediction Model Training - GPU Optimized")
    print("=" * 60)

    # 检查设备
    print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
    print(f"CPU cores available: {len(tf.config.list_physical_devices('CPU'))}")

    # 创建模型
    model = FastTimeModel(model_path="GCM_model_fast.h5")

    # 构建模型
    model.build_model()

    # 训练模型 - 更大的批次大小以充分利用硬件
    history = model.train_model(
        epochs=80,
        batch_size=64,  # 增加批次大小
        validation_split=0.2
    )

    # 评估模型
    results = model.evaluate_model()

    print("\n" + "=" * 60)
    print("Fast training completed successfully!")
    print(f"Model saved as: {model.model_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()