# OmniParser 图像处理服务使用指南

## 环境准备

```bash
# 激活 conda 环境
conda activate omni

# 或使用完整路径
/Users/mac296/miniconda3/envs/omni/bin/python
```

## 快速开始

### 1. 单张图片处理

```python
import asyncio
from omniparser_service import OmniparserService

async def main():
    # 初始化服务（模型只加载一次）
    service = OmniparserService(max_concurrent=4)

    # 处理单张图片
    result = await service.parse("input.png")

    # 返回结果
    # - annotated_image: PIL.Image，带标注的图片
    # - elements: list，结构化元素列表

    # 保存标注图片
    result['annotated_image'].save("output_annotated.png")

    # 查看元素
    for i, elem in enumerate(result['elements']):
        print(f"{i}: {elem['type']} - {elem['content']}")

asyncio.run(main())
```

### 2. 批量并发处理

```python
import asyncio
from omniparser_service import OmniparserService

async def main():
    service = OmniparserService(max_concurrent=4)

    # 批量处理多张图片（并发执行）
    results = await service.parse_batch([
        "image1.png",
        "image2.png",
        "image3.png",
    ])

    # 保存结果
    for i, result in enumerate(results):
        result['annotated_image'].save(f"output_{i}.png")
        print(f"图片 {i+1}: {len(result['elements'])} 个元素")

asyncio.run(main())
```

### 3. 使用示例运行

```bash
# 运行示例脚本
python example_usage.py

# 输出文件
# - output_single.png      # 单张处理结果
# - output_batch_0.png     # 批量处理结果 1
# - output_batch_1.png     # 批量处理结果 2
# - output_batch_2.png     # 批量处理结果 3
```

## 返回数据结构

```python
{
    'annotated_image': PIL.Image,  # 带标注的图片
    'elements': [
        {
            'type': 'text' | 'icon',           # 元素类型
            'bbox': [x1, y1, x2, y2],          # 边界框（比例坐标）
            'interactivity': True | False,      # 是否可交互
            'content': '...',                     # 内容
            'source': '...'                       # 来源
        },
        ...
    ]
}
```

## API 参考

### OmniparserService

```python
class OmniparserService:
    def __init__(self, max_concurrent: int = 4)
    """初始化服务，加载模型（一次性）"""

    async def parse(self, image_path: str) -> dict
    """异步处理单张图片"""

    async def parse_batch(self, image_paths: list) -> list
    """异步批量处理多张图片"""
```

### OmniparserServiceCore

```python
class OmniparserServiceCore:
    def __init__(self, config: dict = None)
    """初始化核心解析器"""

    def parse(self, image_path: str) -> dict
    """同步处理单张图片"""
```

## 配置选项

```python
config = {
    'som_model_path': 'weights/icon_detect/model.pt',      # YOLO 模型路径
    'caption_model_name': 'florence2',                       # Caption 模型名称
    'caption_model_path': 'weights/icon_caption_florence', # Caption 模型路径
    'BOX_TRESHOLD': 0.05,                                  # 检测框阈值
}
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 模型加载 | ~7秒（一次性） |
| 单张处理（MPS） | ~1.5秒 |
| 批量并发 | 2-4张同时处理 |

## 文件结构

```
OmniParser/
├── omniparser_service.py       # 异步并发封装
├── omniparser_service_core.py  # 核心解析器
├── example_usage.py            # 使用示例
└── util/
    ├── omniparser.py          # OmniParser 基类
    └── utils.py               # 工具函数
```

## 注意事项

1. **模型常驻内存**：模型在 `OmniparserService` 初始化时加载一次，之后的调用直接使用
2. **MPS 加速**：Mac Apple Silicon 用户自动使用 MPS 加速
3. **离线模式**：自动设置 `HF_HUB_OFFLINE=1` 加速模型加载
4. **并发控制**：通过 `max_concurrent` 参数控制并发数量
