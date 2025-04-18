# MCP 服务器 用于 基于深度学习的计算机视觉模型

基于 Python 的服务器，实现了模型上下文协议（MCP），用于图像对象检测、分割和姿势估计操作。本仓库基于Ultralytcis和ModelContextProcotol开发
相关链接
Ultralytics - https://github.com/ultralytics/ultralytics
MCP of Python - https://github.com/modelcontextprotocol/python-sdk

## 功能

- 使用 YOLOv10 检测图像中的对象
- 使用 YOLOv8 分割图像中的对象
- 使用 Ultralytics SAM 分割整个图像
- 使用 YOLOv8 估计图像中的人体姿势
- 支持本地和网络图像输入
- 集成 MCP 工具与客户端交互
- 支持 Stdio 和 SSE 传输协议

### TODO
- 支持GroundingDINO
- 支持YOLOE(开放世界检测模型)
- 支持深度估计
- 支持文生图，图生图功能

**注意**：服务器需要有效的图像路径或 URL，并确保以下模型文件可用：`yolov10b.pt`（YOLOv10 检测）、`yolov8n-seg.pt`（YOLOv8 分割）、`yolov8n-pose.pt`（YOLOv8 姿势估计）、`sam_b.pt`（Ultralytics SAM）。

## 快速开始

### 依赖安装
```bash
uv sync
//如需要清华源
uv sync --index https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple

uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 启动服务器

1. **stdio 模式**：

   ```bash
   python server.py
   ```

   输出：

   ```
   使用 stdio 传输启动 MCP 服务器（YOLO）
   ```

2. **SSE 模式**：

   ```bash
   python server.py sse [端口号]
   ```

   示例：

   ```bash
   python server.py sse 8080
   ```

   输出：

   ```
   在端口 8080 上启动 MCP 服务器（YOLO），使用 SSE 传输
   ```


此外，需要下载模型文件在checkpoints文件夹下。
下载链接🔗：https://docs.ultralytics.com/models/yolov10/，https://docs.ultralytics.com/models/yolov8/，https://docs.ultralytics.com/models/sam-2/
下载后目录如下
├── checkpoints
│   ├── sam_b.pt
│   ├── yolov10b.pt
│   ├── yolov8n-pose.pt
│   └── yolov8n-seg.pt

### 模型准备

1. 下载 YOLOv10 预训练模型权重（`yolov10b.pt`）。
2. 确保 `yolov10b.yaml` 配置文件存在，并与代码中指定的路径一致。


## API

### 资源

- `image://system`：图像处理操作接口

### 工具

- **detect_objects**
  - 使用 YOLOv10 检测图像中的对象
  - 输入：`image_url`（字符串）
  - 支持本地路径（`file://` 或相对路径）和网络 URL（`http://` 或 `https://`）
  - 返回 JSON 数组，包含检测到的对象的边界框、置信度和类别标签
  - 示例输出：`[{"box": [x, y, w, h], "confidence": 0.9, "class": "person"}, ...]`
- **segment_objects**
  - 使用 YOLOv8 分割图像中的对象
  - 输入：`image_url`（字符串）
  - 支持本地路径（`file://` 或相对路径）和网络 URL（`http://` 或 `https://`）
  - 返回 JSON 数组，包含分割对象的边界框、置信度和类别标签
  - 示例输出：`[{"box": [x, y, w, h], "confidence": 0.85, "class": "car"}, ...]`
- **segment_image**
  - 使用 Ultralytics SAM 分割整个图像
  - 输入：`image_url`（字符串）
  - 支持本地路径（`file://` 或相对路径）和网络 URL（`http://` 或 `https://`）
  - 返回 JSON 数组，包含分割区域的边界框、面积和置信度
  - 示例输出：`[{"bbox": [x, y, w, h], "area": 2500, "confidence": 0.95}, ...]`
- **estimate_pose**
  - 使用 YOLOv8 估计图像中的人体姿势
  - 输入：`image_url`（字符串）
  - 支持本地路径（`file://` 或相对路径）和网络 URL（`http://` 或 `https://`）
  - 返回 JSON 数组，包含检测到的姿势的关键点坐标和置信度
  - 示例输出：`[{"keypoints": [[x1, y1], [x2, y2], ...], "confidence": [0.9, 0.8, ...]}, ...]`

## 在 Claude Desktop 中使用

在 `claude_desktop_config.json` 中添加以下配置：

**注意**：您可以通过将目录挂载到 `/projects` 为服务器提供沙盒目录。添加 `ro` 标志将使目录对服务器只读。

### SSE

```json
{
  "mcpServers": {
    "server-with-yolo": {
      "url": "http://localhost:8080/sse"
    }
  }
}
```
