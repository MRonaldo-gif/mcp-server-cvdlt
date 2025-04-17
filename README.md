# MCP Server for CVDLT(Computer Vision & Deep Learning Tools)

The repo is based on Ultralytics and Model Context procotol of Python SDK
Related Links:
Ultralytics - https://github.com/ultralytics/ultralytics
MCP of Python - https://github.com/modelcontextprotocol/python-sdk

Python server implementing Model Context Protocol (MCP) for image object detection, segmentation, and pose estimation operations.

## Features

- Detect objects in images using YOLOv10
- Segment objects in images using YOLOv8
- Segment entire images using Ultralytics SAM
- Estimate human poses in images using YOLOv8
- Support for local and network image inputs
- MCP tool integration for client interactions
- Stdio and SSE transport protocols

**Note**: The server requires valid image paths or URLs and access to the following model files: `yolov10b.pt` (YOLOv10 detection), `yolov8n-seg.pt` (YOLOv8 segmentation), `yolov8n-pose.pt` (YOLOv8 pose estimation), and `sam_b.pt` (Ultralytics SAM).

## API

### Resources

- `image://system`: Image processing operations interface

### Tools

- **detect_objects**
  - Detect objects in an image using YOLOv10
  - Input: `image_url` (string)
  - Supports local paths (`file://` or relative) and network URLs (`http://` or `https://`)
  - Returns JSON array of detected objects with bounding boxes, confidence scores, and class labels
  - Example output: `[{"box": [x, y, w, h], "confidence": 0.9, "class": "person"}, ...]`
- **segment_objects**
  - Segment objects in an image using YOLOv8
  - Input: `image_url` (string)
  - Supports local paths (`file://` or relative) and network URLs (`http://` or `https://`)
  - Returns JSON array of segmented objects with bounding boxes, confidence scores, and class labels
  - Example output: `[{"box": [x, y, w, h], "confidence": 0.85, "class": "car"}, ...]`
- **segment_image**
  - Segment entire image using Ultralytics SAM
  - Input: `image_url` (string)
  - Supports local paths (`file://` or relative) and network URLs (`http://` or `https://`)
  - Returns JSON array of segmented regions with bounding boxes, areas, and confidence scores
  - Example output: `[{"bbox": [x, y, w, h], "area": 2500, "confidence": 0.95}, ...]`
- **estimate_pose**
  - Estimate human poses in an image using YOLOv8
  - Input: `image_url` (string)
  - Supports local paths (`file://` or relative) and network URLs (`http://` or `https://`)
  - Returns JSON array of detected poses with keypoint coordinates and confidence scores
  - Example output: `[{"keypoints": [[x1, y1], [x2, y2], ...], "confidence": [0.9, 0.8, ...]}, ...]`

## Usage with Claude Desktop

Add this to your `claude_desktop_config.json`:

**Note**: You can provide sandboxed directories to the server by mounting them to `/projects`. Adding the `ro` flag will make the directory readonly by the server.

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
