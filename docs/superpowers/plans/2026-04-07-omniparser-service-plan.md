# OmniParser 图像处理服务实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建支持模型常驻内存和并发处理的 OmniParser Python 服务

**Architecture:** 双层架构 - `OmniparserServiceCore` 提供同步解析能力，`OmniparserService` 通过 asyncio + ThreadPoolExecutor 实现并发调度。模型在初始化时加载一次，后续调用直接使用。

**Tech Stack:** Python 3.12, asyncio, ThreadPoolExecutor, PyTorch MPS, OmniParser

---

## 文件结构

```
OmniParser/
├── omniparser_service_core.py   # 新增：核心解析器（同步）
├── omniparser_service.py         # 新增：异步服务封装
├── util/
│   ├── omniparser.py            # 现有（设备需更新）
│   └── utils.py                 # 已修复 MPS 支持
```

---

## Task 1: 创建 OmniparserServiceCore

**Files:**
- Create: `omniparser_service_core.py`
- Test: `test_omniparser_service_core.py`

- [ ] **Step 1: 创建测试文件**

```python
# test_omniparser_service_core.py
import pytest
import os

def test_omniparser_service_core_init():
    """测试核心服务初始化"""
    from omniparser_service_core import OmniparserServiceCore

    config = {
        'som_model_path': 'weights/icon_detect/model.pt',
        'caption_model_name': 'florence2',
        'caption_model_path': 'weights/icon_caption_florence',
        'BOX_TRESHOLD': 0.05,
    }
    core = OmniparserServiceCore(config)
    assert core.parser is not None
    print("test_omniparser_service_core_init PASS")

def test_parse_returns_dict():
    """测试 parse 返回正确格式"""
    from omniparser_service_core import OmniparserServiceCore

    config = {
        'som_model_path': 'weights/icon_detect/model.pt',
        'caption_model_name': 'florence2',
        'caption_model_path': 'weights/icon_caption_florence',
        'BOX_TRESHOLD': 0.05,
    }
    core = OmniparserServiceCore(config)
    result = core.parse('imgs/word.png')

    assert 'annotated_image' in result
    assert 'elements' in result
    assert len(result['elements']) > 0
    print("test_parse_returns_dict PASS")

def test_parse_saves_annotated_image(tmp_path):
    """测试保存标注图片"""
    from omniparser_service_core import OmniparserServiceCore
    from PIL import Image

    config = {
        'som_model_path': 'weights/icon_detect/model.pt',
        'caption_model_name': 'florence2',
        'caption_model_path': 'weights/icon_caption_florence',
        'BOX_TRESHOLD': 0.05,
    }
    core = OmniparserServiceCore(config)
    result = core.parse('imgs/word.png')

    output_path = tmp_path / "annotated.png"
    result['annotated_image'].save(str(output_path))

    assert output_path.exists()
    img = Image.open(output_path)
    assert img.size == result['annotated_image'].size
    print("test_parse_saves_annotated_image PASS")
```

- [ ] **Step 2: 运行测试验证失败（模块不存在）**

Run: `cd /Users/mac296/work/OmniParser && python -m pytest test_omniparser_service_core.py::test_omniparser_service_core_init -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 创建 omniparser_service_core.py**

```python
# omniparser_service_core.py
import os
import torch
from PIL import Image
import base64
import io
from typing import Dict

class OmniparserServiceCore:
    """OmniParser 核心解析器，同步处理单张图片"""

    def __init__(self, config: Dict = None):
        """
        初始化核心解析器，加载模型

        Args:
            config: 配置字典，包含:
                - som_model_path: YOLO 模型路径
                - caption_model_name: Caption 模型名称
                - caption_model_path: Caption 模型路径
                - BOX_TRESHOLD: 检测框阈值
                - device: 设备类型 (可选，默认 MPS > CPU)
        """
        if config is None:
            config = {}

        self.config = config

        # 确定设备
        if 'device' in config:
            device_str = config['device']
        else:
            device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device_str) if isinstance(device_str, str) else device_str

        # 设置离线模式加速加载
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

        # 导入并初始化 OmniParser
        from util.omniparser import Omniparser

        som_model_path = config.get('som_model_path', 'weights/icon_detect/model.pt')
        caption_model_name = config.get('caption_model_name', 'florence2')
        caption_model_path = config.get('caption_model_path', 'weights/icon_caption_florence')
        box_threshold = config.get('BOX_TRESHOLD', 0.05)

        parser_config = {
            'som_model_path': som_model_path,
            'caption_model_name': caption_model_name,
            'caption_model_path': caption_model_path,
            'BOX_TRESHOLD': box_threshold,
        }

        # 使用修改后的设备
        self.parser = Omniparser(parser_config)

        # 替换 parser 的设备为 MPS
        self.parser.som_model = self.parser.som_model.to(self.device)
        self.parser.caption_model_processor['model'] = self.parser.caption_model_processor['model'].to(self.device)

        print(f"OmniparserServiceCore initialized on {self.device}")

    def parse(self, image_path: str) -> Dict:
        """
        解析单张图片

        Args:
            image_path: 图片路径

        Returns:
            {
                'annotated_image': PIL.Image,  # 带标注的图片
                'elements': list               # 结构化元素列表
            }
        """
        # 调用底层 parser
        dino_labled_img, parsed_content_list = self.parser.parse_from_file(image_path)

        # 解析 base64 图片
        img_bytes = base64.b64decode(dino_labled_img)
        annotated_image = Image.open(io.BytesIO(img_bytes))

        return {
            'annotated_image': annotated_image,
            'elements': parsed_content_list
        }
```

- [ ] **Step 4: 运行测试验证**

Run: `cd /Users/mac296/work/OmniParser && python -m pytest test_omniparser_service_core.py -v`
Expected: PASS (模型加载需要时间)

- [ ] **Step 5: 提交**

```bash
git add omniparser_service_core.py test_omniparser_service_core.py
git commit -m "feat: add OmniparserServiceCore for single image parsing"
```

---

## Task 2: 更新 Omniparser.parse 方法

**Files:**
- Modify: `util/omniparser.py`

- [ ] **Step 1: 查看现有 parse 方法**

Run: `grep -n "def parse" util/omniparser.py`
Expected: 找到 parse 方法定义

- [ ] **Step 2: 添加 parse_from_file 方法到 Omniparser 类**

```python
# 在 Omniparser 类中添加
def parse_from_file(self, image_path: str):
    """
    从文件路径解析图片

    Args:
        image_path: 图片文件路径

    Returns:
        (dino_labled_img_base64, parsed_content_list)
    """
    from PIL import Image
    image = Image.open(image_path)
    return self.parse_image(image)
```

- [ ] **Step 3: 验证 parse_from_file 存在**

Run: `python -c "from util.omniparser import Omniparser; print(hasattr(Omniparser, 'parse_from_file'))"`
Expected: True

- [ ] **Step 4: 提交**

```bash
git add util/omniparser.py
git commit -m "feat: add parse_from_file method to Omniparser"
```

---

## Task 3: 创建 OmniparserService (异步并发封装)

**Files:**
- Create: `omniparser_service.py`
- Test: `test_omniparser_service.py`

- [ ] **Step 1: 创建测试文件**

```python
# test_omniparser_service.py
import pytest
import asyncio

def test_omniparser_service_init():
    """测试服务初始化"""
    from omniparser_service import OmniparserService

    service = OmniparserService()
    assert service.core is not None
    print("test_omniparser_service_init PASS")

def test_parse_returns_dict():
    """测试异步 parse 返回正确格式"""
    from omniparser_service import OmniparserService

    service = OmniparserService()

    result = service.parse('imgs/word.png')

    assert 'annotated_image' in result
    assert 'elements' in result
    print("test_parse_returns_dict PASS")

@pytest.mark.asyncio
async def test_parse_batch_concurrent():
    """测试批量并发处理"""
    from omniparser_service import OmniparserService

    service = OmniparserService(max_concurrent=2)

    results = await service.parse_batch(['imgs/word.png', 'imgs/word.png'])

    assert len(results) == 2
    assert all('annotated_image' in r for r in results)
    print("test_parse_batch_concurrent PASS")
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd /Users/mac296/work/OmniParser && python -m pytest test_omniparser_service.py::test_omniparser_service_init -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 创建 omniparser_service.py**

```python
# omniparser_service.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import os

from omniparser_service_core import OmniparserServiceCore


class OmniparserService:
    """
    OmniParser 异步并发服务

    支持模型常驻内存，实现 2-4 张图片并发处理
    """

    def __init__(self, max_concurrent: int = 4):
        """
        初始化服务

        Args:
            max_concurrent: 最大并发数，默认 4
        """
        self.max_concurrent = max_concurrent

        # 设置离线模式加速模型加载
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

        # 核心解析器（模型加载一次）
        self.core = OmniparserServiceCore()

        # 线程池用于 CPU-bound 任务
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def parse(self, image_path: str) -> Dict:
        """
        异步解析单张图片

        Args:
            image_path: 图片路径

        Returns:
            {
                'annotated_image': PIL.Image,
                'elements': list
            }
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.core.parse,
            image_path
        )
        return result

    async def parse_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        异步批量解析多张图片

        Args:
            image_paths: 图片路径列表

        Returns:
            结果列表，每个元素包含 annotated_image 和 elements
        """
        tasks = [self.parse(p) for p in image_paths]
        results = await asyncio.gather(*tasks)
        return results

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
```

- [ ] **Step 4: 运行测试验证**

Run: `cd /Users/mac296/work/OmniParser && python -m pytest test_omniparser_service.py -v`
Expected: PASS (需要约 10-20 秒)

- [ ] **Step 5: 提交**

```bash
git add omniparser_service.py test_omniparser_service.py
git commit -m "feat: add OmniparserService with async concurrency support"
```

---

## Task 4: 创建使用示例

**Files:**
- Create: `example_usage.py`

- [ ] **Step 1: 创建示例文件**

```python
# example_usage.py
"""
OmniParser 服务使用示例
"""

import asyncio
from omniparser_service import OmniparserService


async def main():
    # 初始化服务（模型加载一次）
    service = OmniparserService(max_concurrent=4)
    print("服务初始化完成")

    # 单张图片处理
    print("\n处理单张图片...")
    result = await service.parse("imgs/word.png")
    print(f"检测到 {len(result['elements'])} 个元素")
    result['annotated_image'].save("output_single.png")
    print("已保存 output_single.png")

    # 批量并发处理
    print("\n批量并发处理...")
    results = await service.parse_batch([
        "imgs/word.png",
        "imgs/word.png",
        "imgs/word.png",
    ])
    for i, r in enumerate(results):
        print(f"图片 {i+1}: {len(r['elements'])} 个元素")
        r['annotated_image'].save(f"output_batch_{i}.png")
    print("已保存 output_batch_*.png")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 运行示例**

Run: `cd /Users/mac296/work/OmniParser && python example_usage.py`
Expected: 正常运行，生成输出文件

- [ ] **Step 3: 提交**

```bash
git add example_usage.py
git commit -m "docs: add usage example for OmniparserService"
```

---

## Task 5: 最终验证

- [ ] **Step 1: 运行完整测试**

Run: `cd /Users/mac296/work/OmniParser && python -m pytest test_omniparser_service.py test_omniparser_service_core.py -v`
Expected: All PASS

- [ ] **Step 2: 验证生成的文件**

Run: `ls -la *.py output*.png 2>/dev/null || echo "检查输出文件"`
Expected: 看到 example_usage.py 和 output*.png

- [ ] **Step 3: 提交所有更改**

```bash
git add -A
git commit -m "feat: complete Omniparser async service implementation"
```

---

## 总结

| Task | 描述 | 文件 |
|---|---|---|
| 1 | 核心解析器 | `omniparser_service_core.py` |
| 2 | Omniparser parse_from_file | `util/omniparser.py` |
| 3 | 异步并发封装 | `omniparser_service.py` |
| 4 | 使用示例 | `example_usage.py` |
| 5 | 最终验证 | 测试文件 |
