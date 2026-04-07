# OmniParser 图像处理服务设计

## 概述

为 OmniParser 创建本地 Python 调用服务，支持模型常驻内存和并发处理图片。

## 需求

- 本地 Python 直接调用（不使用 HTTP 服务）
- 模型只加载一次，后续调用直接使用
- 支持 2-4 张图片并发处理
- 返回带标注图片 + 结构化元素列表

## 架构

### 核心组件

| 组件 | 职责 |
|---|---|
| `OmniparserServiceCore` | 底层解析器，封装 `Omniparser`，提供同步 `parse()` 方法 |
| `OmniparserService` | 异步调度层，模型常驻内存，管理线程池并发 |

### 文件结构

```
OmniParser/
├── omniparser_service.py      # 新增：服务封装（异步 + 并发）
├── omniparser_service_core.py  # 新增：核心解析器（同步）
├── util/
│   ├── omniparser.py          # 现有类
│   └── utils.py               # 需修复 MPS 支持
```

## API 设计

### OmniparserServiceCore

```python
class OmniparserServiceCore:
    def __init__(self, config: dict = None)
    def parse(self, image_path: str) -> dict:
        """同步解析单张图片
        Returns:
            {
                'annotated_image': PIL.Image,  # 带标注图片
                'elements': list               # 结构化元素列表
            }
        """
```

### OmniparserService

```python
class OmniparserService:
    def __init__(self, max_concurrent: int = 4)
    async def parse(self, image_path: str) -> dict:
        """异步解析单张图片"""
    async def parse_batch(self, image_paths: list) -> list:
        """异步批量解析多张图片"""
```

## 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| som_model_path | weights/icon_detect/model.pt | YOLO 检测模型路径 |
| caption_model_name | florence2 | Caption 模型名称 |
| caption_model_path | weights/icon_caption_florence | Caption 模型路径 |
| BOX_TRESHOLD | 0.05 | 检测框阈值 |
| device | MPS > CPU | 推理设备 |
| max_concurrent | 4 | 最大并发数 |

## 使用示例

```python
import asyncio
from omniparser_service import OmniparserService

async def main():
    service = OmniparserService(max_concurrent=4)

    # 单张处理
    result = await service.parse("test.png")
    result['annotated_image'].save("output.png")

    # 批量并发处理
    results = await service.parse_batch(["img1.png", "img2.png", "img3.png"])

asyncio.run(main())
```

## 性能目标

| 指标 | 目标 |
|---|---|
| 模型加载时间 | < 10s（MPS + 离线模式） |
| 单张处理时间 | < 15s（MPS） |
| 并发支持 | 2-4 张同时处理 |

## 实现任务

1. 创建 `omniparser_service_core.py` - 封装 `Omniparser`，添加 MPS 支持
2. 创建 `omniparser_service.py` - 添加异步调度和并发管理
3. 修复 `util/utils.py` 中的 MPS 设备判断（已部分完成）
4. 添加 `HF_HUB_OFFLINE=1` 环境变量设置以加速模型加载
