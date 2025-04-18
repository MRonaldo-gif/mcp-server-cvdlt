from mcp.server.lowlevel import Server
import mcp.types as types
import os
import sys
import json
import anyio
import urllib.parse
import requests
import tempfile
from ultralytics import YOLO, SAM
import cv2
import numpy as np

# 初始化服务器
app = Server("mcp-server-cvdlt")

def load_image(image_url: str) -> tuple[str, bool]:
    """加载图像，支持本地文件和网络 URL，返回图像路径和是否为临时文件的标志。"""
    parsed_url = urllib.parse.urlparse(image_url)
    if parsed_url.scheme in ("http", "https"):
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code != 200:
            raise Exception(f"无法下载图像，HTTP 状态码 {response.status_code}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            image_path = temp_file.name
        return image_path, True
    else:
        if parsed_url.scheme == "file":
            image_path = urllib.parse.unquote(parsed_url.path)
            if image_path.startswith('/'):
                image_path = image_path[1:]
        else:
            image_path = os.path.abspath(image_url)
        if not os.path.isfile(image_path):
            raise Exception(f"图像文件 {image_path} 不存在。")
        return image_path, False

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """处理客户端的工具调用。"""
    if name == "detect_objects":
        return await detect_objects(arguments)
    elif name == "segment_objects":
        return await segment_objects(arguments)
    elif name == "segment_image":
        return await segment_image(arguments)
    elif name == "estimate_pose":
        return await estimate_pose(arguments)
    return [types.TextContent(type="text", text=f"错误：未知工具：{name}")]

async def detect_objects(arguments: dict) -> list[types.TextContent]:
    """使用 YOLOv10 在图像中检测对象，支持本地文件和网络 URL。"""
    image_url = arguments.get("image_url", "")
    if not image_url:
        return [types.TextContent(type="text", text="错误：未提供图像 URL。")]

    try:
        # 加载 YOLOv10 检测模型
        try:
            yolo_detection = YOLO('checkpoints/yolov10b.pt')
        except Exception as e:
            return [types.TextContent(type="text", text=f"错误：无法加载 YOLOv10 检测模型：{str(e)}")]

        image_path, is_temp = load_image(image_url)
        results = yolo_detection(image_path, conf=0.1)
        class_names = yolo_detection.names
        detections = [
            {"box": box[:4].tolist(), "confidence": float(box[4]), "class": class_names[int(box[5])]}
            for box in results[0].boxes.data
        ]
        if is_temp:
            os.unlink(image_path)
        return [types.TextContent(type="text", text=json.dumps(detections))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{str(e)}")]

async def segment_objects(arguments: dict) -> list[types.TextContent]:
    """使用 YOLOv8 分割图像中的对象，返回边界框和类别。"""
    image_url = arguments.get("image_url", "")
    if not image_url:
        return [types.TextContent(type="text", text="错误：未提供图像 URL。")]

    try:
        # 加载 YOLOv8 分割模型
        try:
            yolo_segment = YOLO('checkpoints/yolov8n-seg.pt')
        except Exception as e:
            return [types.TextContent(type="text", text=f"错误：无法加载 YOLOv8 分割模型：{str(e)}")]

        image_path, is_temp = load_image(image_url)
        results = yolo_segment(image_path)
        detections = []
        for result in results:
            if result.boxes is not None and result.masks is not None:
                for box in result.boxes.data:
                    class_id = int(box[5])
                    class_name = yolo_segment.names[class_id]
                    detections.append({
                        "box": box[:4].tolist(),
                        "confidence": float(box[4]),
                        "class": class_name,
                    })
        if is_temp:
            os.unlink(image_path)
        return [types.TextContent(type="text", text=json.dumps(detections))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{str(e)}")]

async def segment_image(arguments: dict) -> list[types.TextContent]:
    """使用 Ultralytics SAM 对整个图像进行自动分割，返回掩码的边界框和面积。"""
    image_url = arguments.get("image_url", "")
    if not image_url:
        return [types.TextContent(type="text", text="错误：未提供图像 URL。")]

    try:
        # 加载 Ultralytics SAM 模型
        try:
            sam_model = SAM('checkpoints/sam_b.pt')
        except Exception as e:
            return [types.TextContent(type="text", text=f"错误：无法加载 SAM 模型：{str(e)}")]

        image_path, is_temp = load_image(image_url)
        results = sam_model(image_path)
        detections = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    # 将掩码转换为二值图像并计算边界框和面积
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        area = int(cv2.contourArea(contours[0]))
                        detections.append({
                            "bbox": [x, y, w, h],
                            "area": area,
                            "confidence": float(result.boxes.conf[0]) if result.boxes is not None and result.boxes.conf is not None else 1.0
                        })
        if is_temp:
            os.unlink(image_path)
        return [types.TextContent(type="text", text=json.dumps(detections))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{str(e)}")]

async def estimate_pose(arguments: dict) -> list[types.TextContent]:
    """使用 YOLOv8 估计图像中的人体姿势，返回关键点坐标。"""
    image_url = arguments.get("image_url", "")
    if not image_url:
        return [types.TextContent(type="text", text="错误：未提供图像 URL。")]

    try:
        # 加载 YOLOv8 姿势估计模型
        try:
            yolo_pose = YOLO('checkpoints/yolov8n-pose.pt')
        except Exception as e:
            return [types.TextContent(type="text", text=f"错误：无法加载 YOLOv8 姿势估计模型：{str(e)}")]

        image_path, is_temp = load_image(image_url)
        results = yolo_pose(image_path)
        poses = []
        for result in results:
            if result.keypoints is not None:
                for kp in result.keypoints:
                    poses.append({
                        "keypoints": kp.xy.tolist(),
                        "confidence": kp.conf.tolist() if kp.conf is not None else None
                    })
        if is_temp:
            os.unlink(image_path)
        return [types.TextContent(type="text", text=json.dumps(poses))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{str(e)}")]

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """列出可用工具。"""
    return [
        types.Tool(
            name="detect_objects",
            description="使用 YOLOv10 检测图像中的对象",
            inputSchema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "图像文件的 URL（支持 file://、相对路径、http:// 或 https://）"}
                }
            }
        ),
        types.Tool(
            name="segment_objects",
            description="使用 YOLOv8 对图像中的对象进行分割",
            inputSchema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "图像文件的 URL（支持 file://、相对路径、http:// 或 https://）"}
                }
            }
        ),
        types.Tool(
            name="segment_image",
            description="使用 Ultralytics SAM 模型对整个图像进行分割",
            inputSchema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "图像文件的 URL（支持 file://、相对路径、http:// 或 https://）"}
                }
            }
        ),
        types.Tool(
            name="estimate_pose",
            description="使用 YOLOv8 估计图像中的人体姿势",
            inputSchema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "图像文件的 URL（支持 file://、相对路径、http:// 或 https://）"}
                }
            }
        )
    ]

if __name__ == "__main__":
    transport = "stdio"
    port = int(os.getenv("MCP_PORT", "8000"))

    if len(sys.argv) > 1:
        if sys.argv[1] == "sse":
            transport = "sse"
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                print(f"无效的端口号 {sys.argv[2]}，使用端口 {port}")

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        import uvicorn

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ]
        )
        print(f"在端口 {port} 上启动 MCP 服务器（YOLO & SAM），使用 SSE 传输")
        config = uvicorn.Config(app=starlette_app, host="0.0.0.0", port=port)
        server = uvicorn.Server(config)
        import asyncio
        asyncio.run(server.serve())
    else:
        from mcp.server.stdio import stdio_server

        async def run_stdio():
            async with stdio_server() as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())

        print("使用 stdio 传输启动 MCP 服务器（YOLO & SAM）")
        anyio.run(run_stdio)