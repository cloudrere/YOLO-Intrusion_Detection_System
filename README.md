
# YOLO 区域入侵检测系统

基于 PyQt5 + Ultralytics 的桌面检测系统，支持 YOLOv5u / v8 / v9 / v10 / v11 / v12 及自训练权重，覆盖图片、视频、摄像头、RTSP/HLS 视频流四种输入源，内置区域入侵监测、类别筛选、实时统计与 Excel 报表导出。
<img width="1458" height="930" alt="主界面" src="https://github.com/user-attachments/assets/4063076f-caf6-42f4-a45b-fe55697b7392" />
<img width="1458" height="930" alt="选择模型" src="https://github.com/user-attachments/assets/7d38cc65-f39b-491e-b21c-a4c134717eed" />





## 目录结构

```
yolov8\_detect/
├── main.py                 # 主程序入口，UI 逻辑与事件处理
├── ui/
│   ├── \_\_init\_\_.py
│   └── detect\_ui.py        # 界面定义（与业务逻辑分离）
├── utils/
│   ├── \_\_init\_\_.py
│   ├── detector.py         # QThread 推理工作线程
│   ├── line\_draw.py        # 多边形绘制与入侵判定
│   └── excel\_report.py     # Excel 报表生成
├── weights/                # 存放 .pt 模型权重
├── ruqin/                  # 入侵区域 JSON 配置
│   └── ruqin\_example.json
├── output/
│   ├── img\_output/         # 图片检测结果
│   └── video\_output/       # 视频检测结果
├── requirements.txt
└── README.md
```

## 安装

**环境**：Python 3.8+，建议使用独立 conda / venv 环境。

```bash
pip install -r requirements.txt
```

如果遇到 `Failed to initialize NumPy: \_ARRAY\_API not found` 之类的错误，是 NumPy 2.x 与当前 PyTorch 不兼容，强制降级：

```bash
pip install "numpy<2" --force-reinstall
```

如果 requests 警告 urllib3 版本，顺手修掉：

```bash
pip install "urllib3<2" "charset\_normalizer<3.4"
```

GPU 支持需要对应版本的 CUDA 和 PyTorch GPU 版，参考 https://pytorch.org/get-started/locally/ 。

## 运行

```bash
python main.py
```

程序启动后窗口标题栏会显示检测到的设备，例如 `\[GPU: NVIDIA GeForce RTX 4060]` 或 `\[CPU]`。

## 界面布局

三栏布局：

* **左侧功能面板**：模型选择与初始化 → 检测源选择 → 开始/暂停/结束 → 下载结果 → 区域入侵控制
* **中间检测画面**：720×540 固定尺寸显示区，可鼠标单击绘制多边形，右键清空
* **右侧信息面板**：实时统计卡片 → 入侵类别勾选列表 → 详细检测信息文本

## 使用流程

**1. 加载模型**

下拉框内置 `yolov5nu/su/mu`、`yolov8n/s/m/l`、`yolov9t/s/c`、`yolov10n/s/m`、`yolo11n/s/m`、`yolo12n/s`、`best` 等常见权重。若 `weights/` 目录下有同名文件则优先使用本地；否则由 ultralytics 自动下载。点击 **自选模型(.pt)** 可加载任意第三方或自训练权重文件。

选择推理设备（**自动 / GPU / CPU**）。自动模式在 `torch.cuda.is\_available()` 为真时选 GPU，并根据计算能力 ≥7.0 自动启用 FP16。

点击 **初始化模型**，控制台会打印加载耗时和 warmup 耗时。首次 CUDA 推理可能耗时 5\~30 秒，属正常现象。成功后弹窗显示权重文件、类别数、设备信息。

**2. 选择检测源**

三个按钮分别对应三种源，点击后只加载预览、不开始检测：

* **选择图片**：打开本地 jpg/png/jpeg，静态显示
* **选择视频**：打开本地 mp4/avi/mov，显示首帧预览
* **选择摄像头/视频流**：从"视频流选择"下拉框读取源（本地摄像头索引 0 或 RTSP/HLS URL），读一帧作预览后即释放

**3. 可选：设置类别筛选**

右侧"入侵类别"列表在模型加载后自动填充全部类别，默认全部勾选。取消勾选的类别在任何模式下都不会被画框、计数或记录。三个快捷按钮：**全选 / 全不选 / 反选**。

**导入YAML**：支持 YOLO 标准 `data.yaml` 格式，两种 `names` 字段都可识别：

```yaml
# 字典格式
names:
  0: person
  1: bicycle

# 列表格式
names: \['helmet', 'no\_helmet', 'worker']
```

导入后替换当前类别列表。注意类别 ID 需与模型输出索引一致，否则筛选会错位。

**4. 可选：设置入侵区域**

勾选 **区域入侵** 复选框即进入入侵模式。此时必须通过以下方式之一设置多边形：

* **鼠标绘制**：点击 **绘制区域** → 在画面上单击至少 3 个点形成闭合多边形，每单击一次自动保存到 `ruqin/ruqin.json`。右键清除所有点。再次点击按钮变为 **停止绘制**，可退出绘制模式。
* **上传JSON**：点击 **上传区域(JSON)** 选择预先准备的坐标文件。坐标系以检测画面左上角为原点。

```json
{
  "x1": 100, "y1": 100,
  "x2": 500, "y2": 100,
  "x3": 500, "y3": 400,
  "x4": 100, "y4": 400
}
```

若不勾选区域入侵，则跳过本步骤，系统以普通检测模式运行。

**5. 开始检测**

点击醒目的红色 **▶ 开始检测** 按钮。系统会根据当前状态执行：

|场景|行为|
|-|-|
|未勾选"区域入侵"|普通检测模式，对画面内全部勾选类别画框|
|勾选"区域入侵" + 多边形已就绪|入侵检测模式，只对进入多边形的勾选类别画框计数|
|勾选"区域入侵"但没有多边形|提示设置区域或取消勾选，不启动检测|

图片模式为一次性推理；视频/摄像头模式启动独立工作线程，主界面保持响应。

**6. 视频/摄像头控制**

* **暂停/继续**：工作线程暂停读帧，画面定格，统计停止增长
* **结束检测**：停止工作线程、释放摄像头、清除多边形、清空画面

**7. 下载结果**

点击绿色 **下载检测结果** → 选择保存文件夹 → 系统输出两个同名文件：

* 媒体文件：图片（jpg/png）或视频（mp4）
* Excel 报表：`<名称>\_report.xlsx`

## 实时统计面板

|字段|颜色|含义|何时有值|
|-|-|-|-|
|**当前入侵数**（选中类别）|红|当前帧位于多边形内的勾选类别目标数|区域入侵模式|
|**非入侵检测数**（选中类别）|绿|当前帧检测到的勾选类别目标总数|普通检测模式|
|**总处理帧数**|蓝|工作线程累计处理的帧数|始终|

两个计数互斥——当前模式下对应字段有值，另一个归 0。

## Excel 报表

三个 Sheet：

* **Summary**：源类型、源路径、模型文件、类别总数、筛选类别、推理设备、FP16 状态、区域入侵开关、多边形点数、起止时间、耗时、总处理帧数、入侵次数、检测到目标总数、输出路径
* **DetectionLog**：逐目标记录，包含帧序号、时间戳、类别ID、类别名、置信度、边界框 X1/Y1/X2/Y2、是否在区域内
* **ClassSummary**：按类别聚合的出现次数与区域内次数

日志上限 20000 条，超出后不再追加，长视频不会撑爆内存。

## 架构说明

**UI 与逻辑分离**：`ui/detect\_ui.py` 仅包含 `Ui\_MainWindow` 布局和样式，所有业务逻辑集中在 `main.py` 的 `UI\_Logic\_Window` 类中。

**多线程推理**：`utils/detector.py` 的 `DetectionWorker(QThread)` 在独立线程完成视频读取、YOLO 推理、视频写盘，通过 `pyqtSignal` 把处理后的帧和统计数据传回主线程。所有共享状态（停止标志、暂停标志、多边形、类别筛选）由 `QMutex` 保护。UI 主线程只负责渲染，不会被推理阻塞。

**FP16 稳健性**：自动模式下只在 GPU 计算能力 ≥7.0 时启用 FP16。若 warmup 或推理过程中 FP16 抛异常，自动降级到 FP32 重试，不会因单次精度问题整体崩溃。

**坐标缩放**：原始图像 resize 到 720×540 显示，全局 `o\_n\_x\_scale`/`o\_n\_y\_scale` 把画布坐标映射回原始分辨率，保证鼠标绘制的多边形在高/低分辨率源上位置一致。

## 常见问题

**启动时卡在"warmup 开始"不动**
首次 CUDA 初始化正常需要 5\~30 秒，耐心等。如果超过 1 分钟，查看控制台是否有异常栈。

**`Failed to initialize NumPy: \_ARRAY\_API not found`**
降级 numpy：`pip install "numpy<2" --force-reinstall`。

**摄像头/RTSP 打开失败**
检查摄像头占用（其他程序是否正在使用）、URL 是否可访问。RTSP 通常比 HLS 延迟更低但对网络更敏感。

**绘制多边形位置有偏移**
确保没有手动调整画面显示大小，`MyLabel` 已通过 `setFixedSize(720, 540)` 锁定。

**Excel 下载按钮显示"无可下载结果"**
必须先完成（或至少运行过）一次检测才会生成输出文件。

## 依赖清单

```
ultralytics>=8.0.0     # YOLO 推理框架
PyQt5>=5.15.0          # GUI
opencv-python>=4.5.0   # 视频读写、图像处理
numpy<2                # 与当前 PyTorch 兼容
torch>=1.8.0           # 深度学习后端
openpyxl>=3.0.0        # Excel 报表
pyyaml>=5.4.0          # YAML 类别导入
```

## 许可与致谢

本项目为学习用途。YOLO 权重遵循 Ultralytics 官方许可；UI 框架基于 Qt / PyQt5。

