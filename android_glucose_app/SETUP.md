# 🔧 Android应用设置指南

## 步骤 1: 复制TFLite模型

首先，需要将TFLite模型复制到Android项目的assets目录：

```bash
# 在项目根目录执行
cd /home/gitlab-runner/2024_TJU_Data_Mining-Analysis

# 复制TFLite模型
cp mobile_deployment/mobile_deployment/src/models/glucose_predictor.tflite \
   android_glucose_app/app/src/main/assets/
```

## 步骤 2: 验证模型文件

```bash
# 检查模型是否存在
ls -lh android_glucose_app/app/src/main/assets/glucose_predictor.tflite

# 应该看到类似输出:
# -rw-r--r-- 1 user group 202K glucose_predictor.tflite
```

## 步骤 3: 在Android Studio中打开项目

1. 启动 Android Studio
2. 选择 **File** > **Open**
3. 导航到 `android_glucose_app` 目录
4. 点击 **OK**

## 步骤 4: 等待Gradle同步

Android Studio会自动：
- 下载所有依赖库
- 配置项目
- 索引代码

这可能需要几分钟，请耐心等待。

## 步骤 5: 配置运行设备

### 选项A: 使用真实设备

1. 在手机上启用开发者选项：
   - 进入 **设置** > **关于手机**
   - 连续点击 **版本号** 7次

2. 启用USB调试：
   - **设置** > **开发者选项** > **USB调试**

3. 连接手机到电脑
4. 允许USB调试授权

### 选项B: 使用Android模拟器

1. 在Android Studio中打开 **AVD Manager**
2. 点击 **Create Virtual Device**
3. 选择设备型号（推荐: Pixel 5）
4. 选择系统镜像（推荐: Android 11 或更高）
5. 完成创建并启动模拟器

## 步骤 6: 运行应用

1. 在Android Studio顶部工具栏选择目标设备
2. 点击绿色的运行按钮 ▶️
3. 或者按快捷键 `Shift + F10`

## 🎯 预期结果

应用启动后，您应该看到：

1. ✅ 患者信息卡片（Patient 2035_0_20210629）
2. ✅ 血糖预测图表（5条线）
3. ✅ 预测结果详情（文本显示）
4. ✅ "重新预测"按钮

## 🐛 常见问题

### 问题 1: "glucose_predictor.tflite not found"

**解决方案**:
```bash
# 确保模型文件在正确位置
ls android_glucose_app/app/src/main/assets/glucose_predictor.tflite

# 如果不存在，执行步骤1复制模型
```

### 问题 2: Gradle同步失败

**解决方案**:
1. 检查网络连接
2. 在Android Studio中: **File** > **Invalidate Caches / Restart**
3. 重新同步: **File** > **Sync Project with Gradle Files**

### 问题 3: 编译错误 "Unresolved reference: MPAndroidChart"

**解决方案**:
1. 确保 `settings.gradle` 中包含 JitPack 仓库:
   ```gradle
   maven { url 'https://jitpack.io' }
   ```
2. 重新同步Gradle

### 问题 4: 应用崩溃 "TensorFlow Lite model not found"

**解决方案**:
1. 检查 `app/build.gradle` 中的 `aaptOptions`:
   ```gradle
   aaptOptions {
       noCompress "tflite"
   }
   ```
2. Clean & Rebuild: **Build** > **Clean Project** 然后 **Build** > **Rebuild Project**

### 问题 5: 图表不显示

**解决方案**:
1. 检查设备API级别 >= 24
2. 查看Logcat日志: `adb logcat -s MainActivity`
3. 确保预测数据不为空

## 📝 开发建议

### 1. 启用日志输出

在 `MainActivity.kt` 和 `GlucosePredictor.kt` 中，所有日志都使用 `Log.d()` 输出。

查看日志:
```bash
adb logcat | grep -E "(GlucosePredictor|MainActivity)"
```

### 2. 调试断点

在以下位置设置断点进行调试：
- `GlucosePredictor.predict()` - 查看预测输入
- `MainActivity.updateChart()` - 查看图表数据
- `MainActivity.displayResults()` - 查看结果格式化

### 3. 修改患者数据

在 `MainActivity.getSamplePatientData()` 中修改示例数据：

```kotlin
private fun getSamplePatientData(): PatientData {
    // 修改历史血糖值
    val historicalGlucose = floatArrayOf(
        142.2f, 158.4f, 172.8f, // ... 你的数据
    )
    // ...
}
```

## 🎨 自定义样式

### 修改颜色主题

编辑 `res/values/colors.xml`:
```xml
<color name="purple_500">#FF6200EE</color>  <!-- 按钮颜色 -->
<color name="purple_700">#FF3700B3</color>  <!-- 标题颜色 -->
```

### 修改图表样式

在 `MainActivity.setupChart()` 中：
```kotlin
chart.apply {
    description.textSize = 14f  // 描述文字大小
    legend.textSize = 10f       // 图例文字大小
    // ...
}
```

## 📦 打包APK

### Debug APK (开发版本)
```bash
cd android_glucose_app
./gradlew assembleDebug

# APK位置: app/build/outputs/apk/debug/app-debug.apk
```

### Release APK (发布版本)
```bash
./gradlew assembleRelease

# 需要签名配置
# APK位置: app/build/outputs/apk/release/app-release.apk
```

## 🚀 性能分析

### 测量推理时间

在 `GlucosePredictor.predict()` 中添加：
```kotlin
val startTime = System.currentTimeMillis()
interpreter?.runForMultipleInputsOutputs(inputs, outputs)
val inferenceTime = System.currentTimeMillis() - startTime
Log.d(tag, "推理时间: ${inferenceTime}ms")
```

### 内存分析

在Android Studio中：
1. **View** > **Tool Windows** > **Profiler**
2. 选择你的应用进程
3. 点击 **Memory** 查看内存使用

## ✅ 验证清单

设置完成后，请验证：

- [ ] TFLite模型文件已复制到 assets 目录
- [ ] Gradle同步成功，无错误
- [ ] 应用可以成功安装到设备/模拟器
- [ ] 应用启动无崩溃
- [ ] 图表正确显示5条线（历史+4条预测）
- [ ] 预测结果文本正确显示
- [ ] "重新预测"按钮可点击并刷新数据

## 📞 获取帮助

如果遇到其他问题：

1. 查看完整日志: `adb logcat > logcat.txt`
2. 检查 `README.md` 了解更多细节
3. 参考 `demo3_tflite_model.py` 的Python实现

---

**准备好了吗？** 返回 [README.md](README.md) 查看完整文档！
