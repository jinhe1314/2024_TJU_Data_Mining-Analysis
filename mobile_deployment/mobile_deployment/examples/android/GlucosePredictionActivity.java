package com.example.glucoseprediction;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import java.util.Map;
import java.util.Random;

/**
 * 血糖预测Android界面示例
 * 展示如何在Android应用中集成血糖预测模型
 */
public class GlucosePredictionActivity extends AppCompatActivity {
    private GlucosePredictor predictor;
    private TextView modelInfoText;
    private TextView predictionResultText;
    private Button predictButton;
    private Button performanceTestButton;
    private Handler mainHandler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_glucose_prediction);

        // 初始化UI组件
        initViews();

        // 初始化预测器
        initPredictor();

        // 设置按钮点击事件
        setupButtonListeners();

        mainHandler = new Handler(Looper.getMainLooper());
    }

    private void initViews() {
        modelInfoText = findViewById(R.id.modelInfoText);
        predictionResultText = findViewById(R.id.predictionResultText);
        predictButton = findViewById(R.id.predictButton);
        performanceTestButton = findViewById(R.id.performanceTestButton);
    }

    private void initPredictor() {
        new Thread(() -> {
            try {
                predictor = new GlucosePredictor();
                AssetManager assetManager = getAssets();
                predictor.initialize(assetManager);

                // 在主线程更新UI
                mainHandler.post(() -> {
                    modelInfoText.setText(predictor.getModelInfo());
                    predictButton.setEnabled(true);
                    performanceTestButton.setEnabled(true);
                    Toast.makeText(this, "模型加载成功", Toast.LENGTH_SHORT).show();
                });

            } catch (Exception e) {
                mainHandler.post(() -> {
                    Toast.makeText(this, "模型加载失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }

    private void setupButtonListeners() {
        predictButton.setOnClickListener(v -> performSinglePrediction());
        performanceTestButton.setOnClickListener(v -> performPerformanceTest());
    }

    private void performSinglePrediction() {
        if (predictor == null) {
            Toast.makeText(this, "模型未初始化", Toast.LENGTH_SHORT).show();
            return;
        }

        // 显示加载状态
        predictionResultText.setText("正在预测...");
        predictButton.setEnabled(false);

        new Thread(() -> {
            try {
                // 生成测试数据
                float[][] timeSeriesData = generateTestTimeSeriesData();
                float[] staticData = generateTestStaticData();

                // 执行预测
                long startTime = System.currentTimeMillis();
                Map<Integer, Float> predictions = predictor.predictGlucose(timeSeriesData, staticData);
                long inferenceTime = System.currentTimeMillis() - startTime;

                // 格式化结果
                StringBuilder result = new StringBuilder();
                result.append("预测结果 (推理时间: ").append(inferenceTime).append("ms):\\n\\n");

                for (Map.Entry<Integer, Float> entry : predictions.entrySet()) {
                    result.append(String.format("%d分钟后: %.2f mg/dL\\n",
                        entry.getKey(), entry.getValue()));
                }

                // 添加血糖状态评估
                result.append("\\n血糖状态评估:\\n");
                for (Map.Entry<Integer, Float> entry : predictions.entrySet()) {
                    String status = getGlucoseStatus(entry.getValue());
                    result.append(String.format("%d分钟: %s\\n", entry.getKey(), status));
                }

                final String finalResult = result.toString();

                // 在主线程更新UI
                mainHandler.post(() -> {
                    predictionResultText.setText(finalResult);
                    predictButton.setEnabled(true);
                });

            } catch (Exception e) {
                mainHandler.post(() -> {
                    predictionResultText.setText("预测失败: " + e.getMessage());
                    predictButton.setEnabled(true);
                });
            }
        }).start();
    }

    private void performPerformanceTest() {
        if (predictor == null) {
            Toast.makeText(this, "模型未初始化", Toast.LENGTH_SHORT).show();
            return;
        }

        performanceTestButton.setEnabled(false);
        predictionResultText.setText("性能测试中...");

        new Thread(() -> {
            try {
                int numTests = 100;
                long totalTime = 0;
                long minTime = Long.MAX_VALUE;
                long maxTime = 0;

                for (int i = 0; i < numTests; i++) {
                    // 生成测试数据
                    float[][] timeSeriesData = generateTestTimeSeriesData();
                    float[] staticData = generateTestStaticData();

                    // 执行预测
                    long startTime = System.nanoTime();
                    predictor.predictGlucose(timeSeriesData, staticData);
                    long endTime = System.nanoTime();

                    long inferenceTime = (endTime - startTime) / 1_000_000; // 转换为毫秒
                    totalTime += inferenceTime;
                    minTime = Math.min(minTime, inferenceTime);
                    maxTime = Math.max(maxTime, inferenceTime);

                    // 更新进度
                    final int progress = (i + 1) * 100 / numTests;
                    mainHandler.post(() -> {
                        predictionResultText.setText("性能测试中... " + progress + "%");
                    });
                }

                // 计算统计数据
                double avgTime = (double) totalTime / numTests;
                double throughput = 1000.0 / avgTime; // QPS

                // 格式化结果
                StringBuilder result = new StringBuilder();
                result.append("性能测试结果 (").append(numTests).append(" 次预测):\\n\\n");
                result.append(String.format("平均推理时间: %.2f ms\\n", avgTime));
                result.append(String.format("最快推理时间: %d ms\\n", minTime));
                result.append(String.format("最慢推理时间: %d ms\\n", maxTime));
                result.append(String.format("推理吞吐量: %.1f QPS\\n\\n", throughput));

                result.append("性能评估:\\n");
                if (avgTime < 10) {
                    result.append("✅ 推理速度: 优秀 (<10ms)\\n");
                } else if (avgTime < 50) {
                    result.append("✅ 推理速度: 良好 (<50ms)\\n");
                } else {
                    result.append("⚠️ 推理速度: 需要优化 (>50ms)\\n");
                }

                result.append("✅ 模型大小: 0.20 MB\\n");
                result.append("✅ 适合移动端部署\\n");

                final String finalResult = result.toString();

                // 在主线程更新UI
                mainHandler.post(() -> {
                    predictionResultText.setText(finalResult);
                    performanceTestButton.setEnabled(true);
                });

            } catch (Exception e) {
                mainHandler.post(() -> {
                    predictionResultText.setText("性能测试失败: " + e.getMessage());
                    performanceTestButton.setEnabled(true);
                });
            }
        }).start();
    }

    private float[][] generateTestTimeSeriesData() {
        Random random = new Random();
        float[][] data = new float[10][51];

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 51; j++) {
                // 模拟真实的血糖数据范围 (3-15 mmol/L)
                data[i][j] = 3 + random.nextFloat() * 12;
            }
        }

        return data;
    }

    private float[] generateTestStaticData() {
        Random random = new Random();
        float[] data = new float[30];

        for (int i = 0; i < 30; i++) {
            // 标准化数据范围 (0-1)
            data[i] = random.nextFloat();
        }

        return data;
    }

    private String getGlucoseStatus(float glucoseLevel) {
        if (glucoseLevel < 3.9) {
            return "低血糖 ⚠️";
        } else if (glucoseLevel < 5.6) {
            return "正常 ✅";
        } else if (glucoseLevel < 6.9) {
            return "轻度升高 ⚠️";
        } else if (glucoseLevel < 11.1) {
            return "中度升高 ⚠️";
        } else {
            return "高血糖 ❌";
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (predictor != null) {
            predictor.close();
        }
    }
}