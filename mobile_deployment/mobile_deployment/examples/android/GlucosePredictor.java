package com.example.glucoseprediction;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

/**
 * 血糖预测模型 - Android TensorFlow Lite 集成示例
 * 基于LSTM + Cross-Attention架构的血糖水平预测模型
 *
 * 功能：
 * - 预测未来15、30、45、60分钟的血糖水平
 * - 输入：10个时间步的历史数据(51特征) + 静态患者特征(30特征)
 * - 输出：4个时间点的血糖预测值
 */
public class GlucosePredictor {
    private static final String MODEL_FILE = "glucose_predictor.tflite";
    private static final int TIME_SERIES_LENGTH = 10;
    private static final int TIME_SERIES_FEATURES = 51;
    private static final int STATIC_FEATURES = 30;
    private static final int OUTPUT_LENGTH = 4;

    private Interpreter interpreter;
    private boolean isInitialized = false;

    // 预测时间点（分钟）
    private static final int[] PREDICTION_HORIZONS = {15, 30, 45, 60};

    /**
     * 初始化模型
     * @param assetManager Android资源管理器
     */
    public void initialize(AssetManager assetManager) throws IOException {
        if (isInitialized) {
            return;
        }

        // 加载TFLite模型
        MappedByteBuffer modelBuffer = loadModelFile(assetManager, MODEL_FILE);

        // 创建解释器选项
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4); // 使用4个线程进行推理

        // 创建解释器
        this.interpreter = new Interpreter(modelBuffer, options);
        this.isInitialized = true;

        // 输出模型信息
        logModelInfo();
    }

    /**
     * 预测血糖水平
     * @param timeSeriesData 10个时间步的历史数据 [10][51]
     * @param staticData 静态患者特征 [30]
     * @return 预测结果 Map<时间点(分钟), 血糖预测值>
     */
    public Map<Integer, Float> predictGlucose(float[][] timeSeriesData, float[] staticData) {
        if (!isInitialized) {
            throw new IllegalStateException("模型未初始化，请先调用initialize()");
        }

        // 验证输入数据维度
        validateInputData(timeSeriesData, staticData);

        // 准备输入缓冲区
        ByteBuffer timeSeriesBuffer = ByteBuffer.allocateDirect(
            TIME_SERIES_LENGTH * TIME_SERIES_FEATURES * Float.BYTES);
        timeSeriesBuffer.order(ByteOrder.nativeOrder());

        ByteBuffer staticBuffer = ByteBuffer.allocateDirect(
            STATIC_FEATURES * Float.BYTES);
        staticBuffer.order(ByteOrder.nativeOrder());

        // 填充时序数据
        for (int i = 0; i < TIME_SERIES_LENGTH; i++) {
            for (int j = 0; j < TIME_SERIES_FEATURES; j++) {
                timeSeriesBuffer.putFloat(timeSeriesData[i][j]);
            }
        }
        timeSeriesBuffer.rewind();

        // 填充静态数据
        for (int i = 0; i < STATIC_FEATURES; i++) {
            staticBuffer.putFloat(staticData[i]);
        }
        staticBuffer.rewind();

        // 准备输出缓冲区
        float[][] outputData = new float[1][OUTPUT_LENGTH];

        // 执行推理
        Object[] inputs = {timeSeriesBuffer, staticBuffer};
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputData);

        interpreter.runForMultipleInputsOutputs(inputs, outputs);

        // 处理输出结果
        Map<Integer, Float> predictions = new HashMap<>();
        float[] rawPredictions = outputData[0];

        for (int i = 0; i < OUTPUT_LENGTH; i++) {
            predictions.put(PREDICTION_HORIZONS[i], rawPredictions[i]);
        }

        return predictions;
    }

    /**
     * 批量预测（用于性能测试）
     * @param batchData 批量输入数据
     * @return 批量预测结果
     */
    public Map<Integer, Float>[] predictBatch(float[][][] batchTimeSeriesData, float[][] batchStaticData) {
        if (!isInitialized) {
            throw new IllegalStateException("模型未初始化，请先调用initialize()");
        }

        int batchSize = batchTimeSeriesData.length;
        Map<Integer, Float>[] batchResults = new HashMap[batchSize];

        for (int i = 0; i < batchSize; i++) {
            batchResults[i] = predictGlucose(batchTimeSeriesData[i], batchStaticData[i]);
        }

        return batchResults;
    }

    /**
     * 获取模型信息
     */
    public String getModelInfo() {
        if (!isInitialized) {
            return "模型未初始化";
        }

        StringBuilder info = new StringBuilder();
        info.append("血糖预测模型信息:\\n");
        info.append("- 模型类型: LSTM + Cross-Attention\\n");
        info.append("- 输入: 时序数据[10,51] + 静态特征[30]\\n");
        info.append("- 输出: 血糖预测[4] (15/30/45/60分钟)\\n");

        // 获取输入输出张量信息
        Tensor[] inputTensors = interpreter.getInputTensors();
        Tensor[] outputTensors = interpreter.getOutputTensors();

        info.append(String.format("- 输入数量: %d\\n", inputTensors.length));
        for (int i = 0; i < inputTensors.length; i++) {
            Tensor tensor = inputTensors[i];
            info.append(String.format("  输入%d: %s (%s)\\n",
                i + 1,
                shapeToString(tensor.shape()),
                tensor.dataType()));
        }

        info.append(String.format("- 输出数量: %d\\n", outputTensors.length));
        for (int i = 0; i < outputTensors.length; i++) {
            Tensor tensor = outputTensors[i];
            info.append(String.format("  输出%d: %s (%s)\\n",
                i + 1,
                shapeToString(tensor.shape()),
                tensor.dataType()));
        }

        return info.toString();
    }

    /**
     * 释放资源
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            isInitialized = false;
        }
    }

    // ========== 私有方法 ==========

    /**
     * 从Assets加载模型文件
     */
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * 验证输入数据维度
     */
    private void validateInputData(float[][] timeSeriesData, float[] staticData) {
        if (timeSeriesData.length != TIME_SERIES_LENGTH) {
            throw new IllegalArgumentException(
                String.format("时序数据长度错误，期望%d，实际%d",
                    TIME_SERIES_LENGTH, timeSeriesData.length));
        }

        for (int i = 0; i < timeSeriesData.length; i++) {
            if (timeSeriesData[i].length != TIME_SERIES_FEATURES) {
                throw new IllegalArgumentException(
                    String.format("时序数据特征数错误，期望%d，实际%d",
                        TIME_SERIES_FEATURES, timeSeriesData[i].length));
            }
        }

        if (staticData.length != STATIC_FEATURES) {
            throw new IllegalArgumentException(
                String.format("静态特征数错误，期望%d，实际%d",
                    STATIC_FEATURES, staticData.length));
        }
    }

    /**
     * 将张量形状转换为字符串
     */
    private String shapeToString(int[] shape) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * 输出模型信息
     */
    private void logModelInfo() {
        System.out.println("血糖预测模型初始化成功");
        System.out.println(getModelInfo());
    }

    // ========== 使用示例 ==========

    /**
     * 使用示例代码
     */
    public static void usageExample() {
        // 初始化预测器
        GlucosePredictor predictor = new GlucosePredictor();
        try {
            // AssetManager assetManager = context.getAssets();
            // predictor.initialize(assetManager);

            // 准备测试数据
            float[][] timeSeriesData = new float[10][51];
            float[] staticData = new float[30];

            // 填充数据（这里使用随机数据作为示例）
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 51; j++) {
                    timeSeriesData[i][j] = (float) Math.random();
                }
            }

            for (int i = 0; i < 30; i++) {
                staticData[i] = (float) Math.random();
            }

            // 执行预测
            Map<Integer, Float> predictions = predictor.predictGlucose(timeSeriesData, staticData);

            // 输出结果
            System.out.println("血糖预测结果:");
            for (Map.Entry<Integer, Float> entry : predictions.entrySet()) {
                System.out.printf("%d分钟后: %.2f mg/dL\\n", entry.getKey(), entry.getValue());
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            predictor.close();
        }
    }
}