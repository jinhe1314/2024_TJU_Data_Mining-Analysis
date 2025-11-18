#!/usr/bin/env python3
"""
血糖预测模型训练脚本 - TensorFlow 2.13优化版本
Blood Glucose Prediction Model Training Script - Optimized for TensorFlow 2.13
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
from typing import Tuple, Dict, Any

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

class EnhancedTimeModel:
    """
    增强版时间序列模型，专门针对TensorFlow 2.13优化
    Enhanced Time Series Model optimized for TensorFlow 2.13
    """

    def __init__(self,
                 num_units: int = 64,
                 model_path: str = "GCM_model_tf213.h5",
                 learning_rate: float = 0.001,
                 dropout_rate: float = 0.3):
        self.scaler_ts_X = StandardScaler()
        self.scaler_static_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.num_units = num_units
        self.model_save_path = model_path
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # 加载配置
        self._load_attributes()

        # 加载数据
        self._load_data()

        # 创建序列
        self._create_sequences()

        # 数据标准化
        self._standardize_data()

        # 分割数据
        self._split_data()

        self.model = None
        self.history = None

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

    def build_model(self) -> Model:
        """
        构建增强的混合神经网络模型
        Build enhanced hybrid neural network model
        """
        print("Building enhanced model...")

        # 时序数据输入
        ts_input = Input(shape=(self.ts_X_train.shape[1], self.ts_X_train.shape[2]),
                        name='timeseries_input')

        # LSTM编码器 - 添加BatchNormalization和Dropout
        x = LSTM(64, return_sequences=True, name='lstm_1')(ts_input)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = LSTM(56, return_sequences=True, name='lstm_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = LSTM(48, return_sequences=True, name='lstm_3')(x)
        x = BatchNormalization()(x)

        x = LSTM(40, return_sequences=True, name='lstm_4')(x)
        x = BatchNormalization()(x)

        x = LSTM(36, return_sequences=True, name='lstm_5')(x)
        x = BatchNormalization()(x)

        x = LSTM(32, return_sequences=False, name='lstm_6')(x)
        x = BatchNormalization()(x)
        ts_embedding = x

        # 静态特征输入
        static_input = Input(shape=(self.static_X_train.shape[1],),
                           name='static_input')

        # Dense编码器 - 添加BatchNormalization和Dropout
        y = Dense(64, activation='relu', name='dense_static_1')(static_input)
        y = BatchNormalization()(y)
        y = Dropout(self.dropout_rate)(y)

        y = Dense(56, activation='relu', name='dense_static_2')(y)
        y = BatchNormalization()(y)
        y = Dropout(self.dropout_rate)(y)

        y = Dense(48, activation='relu', name='dense_static_3')(y)
        y = BatchNormalization()(y)

        y = Dense(40, activation='relu', name='dense_static_4')(y)
        y = BatchNormalization()(y)

        y = Dense(36, activation='relu', name='dense_static_5')(y)
        y = BatchNormalization()(y)

        y = Dense(32, activation='relu', name='dense_static_6')(y)
        static_embedding = y

        # Cross-Attention机制
        def cross_attention(query, key, value, name=''):
            attention_scores = tf.matmul(query, key, transpose_b=True)
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            attended_vector = tf.matmul(attention_weights, value)
            return attended_vector

        # 静态特征查询时序特征
        query1 = tf.expand_dims(static_embedding, axis=1)
        key1 = tf.expand_dims(ts_embedding, axis=1)
        value1 = tf.expand_dims(ts_embedding, axis=1)
        cross_attention_output1 = cross_attention(query1, key1, value1, 'static_to_ts')
        cross_attention_output1 = Flatten()(cross_attention_output1)

        # 时序特征查询静态特征
        query2 = tf.expand_dims(ts_embedding, axis=1)
        key2 = tf.expand_dims(static_embedding, axis=1)
        value2 = tf.expand_dims(static_embedding, axis=1)
        cross_attention_output2 = cross_attention(query2, key2, value2, 'ts_to_static')
        cross_attention_output2 = Flatten()(cross_attention_output2)

        # 合并attention输出
        merged_attention = Concatenate(name='merge_attention')(
            [cross_attention_output1, cross_attention_output2]
        )

        # 解码器 - 逐渐减少维度
        z = Dense(64, activation='relu', name='decoder_1')(merged_attention)
        z = BatchNormalization()(z)
        z = Dropout(self.dropout_rate)(z)

        z = Dense(56, activation='relu', name='decoder_2')(z)
        z = BatchNormalization()(z)
        z = Dropout(self.dropout_rate)(z)

        z = Dense(48, activation='relu', name='decoder_3')(z)
        z = BatchNormalization()(z)

        z = Dense(40, activation='relu', name='decoder_4')(z)
        z = BatchNormalization()(z)

        z = Dense(36, activation='relu', name='decoder_5')(z)
        z = BatchNormalization()(z)

        z = Dense(32, activation='relu', name='decoder_6')(z)
        z = Dropout(self.dropout_rate)(z)

        z = Dense(16, activation='relu', name='decoder_7')(z)
        output = Dense(4, name='output')(z)  # 4个目标值

        # 构建模型
        self.model = Model(
            inputs=[ts_input, static_input],
            outputs=output,
            name='BloodGlucosePredictionModel_TF213'
        )

        # 编译模型 - 使用更好的优化器和学习率
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        # 打印模型摘要
        self.model.summary()

        return self.model

    def train_model(self,
                   epochs: int = 100,
                   batch_size: int = 32,
                   validation_split: float = 0.2,
                   patience_early_stop: int = 10,
                   patience_lr_reduce: int = 5) -> Dict[str, Any]:
        """
        训练模型，添加早停和学习率调整
        Train model with early stopping and learning rate scheduling
        """
        print(f"Training model for {epochs} epochs with batch size {batch_size}...")

        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience_early_stop,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience_lr_reduce,
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

    def evaluate_model(self) -> Dict[str, float]:
        """评估模型性能"""
        print("Evaluating model...")

        test_loss, test_mae = self.model.evaluate(
            [self.ts_X_test, self.static_X_test],
            self.y_test,
            verbose=0
        )

        # 预测并计算额外的指标
        y_pred = self.model.predict([self.ts_X_test, self.static_X_test])

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

    def save_training_log(self, history: Dict, results: Dict):
        """保存训练日志"""
        with open("training_log_tf213.log", 'w', encoding='utf-8') as file:
            file.write(f"TensorFlow Version: {tf.__version__}\n")
            file.write(f"Training Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Model Parameters: num_units={self.num_units}, "
                      f"lr={self.learning_rate}, dropout={self.dropout_rate}\n\n")

            file.write("Training History:\n")
            for epoch, (loss, mae, val_loss, val_mae) in enumerate(zip(
                history['loss'], history['mean_absolute_error'],
                history['val_loss'], history['val_mean_absolute_error']
            )):
                file.write(f"Epoch {epoch + 1}: "
                          f"Loss={loss:.4f}, MAE={mae:.4f}, "
                          f"Val_Loss={val_loss:.4f}, Val_MAE={val_mae:.4f}\n")

            file.write(f"\nEvaluation Results:\n")
            for key, value in results.items():
                file.write(f"{key}: {value:.4f}\n")

def main():
    """主函数"""
    print("=" * 60)
    print("血糖预测模型训练 - TensorFlow 2.13")
    print("Blood Glucose Prediction Model Training - TensorFlow 2.13")
    print("=" * 60)

    # 创建模型实例
    model = EnhancedTimeModel(
        num_units=64,
        model_path="GCM_model_tf213.h5",
        learning_rate=0.001,
        dropout_rate=0.3
    )

    # 构建模型
    model.build_model()

    # 训练模型
    history = model.train_model(
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        patience_early_stop=15,
        patience_lr_reduce=7
    )

    # 评估模型
    results = model.evaluate_model()

    # 保存训练日志
    model.save_training_log(history, results)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved as: {model.model_save_path}")
    print("Training log saved as: training_log_tf213.log")
    print("=" * 60)

if __name__ == "__main__":
    main()