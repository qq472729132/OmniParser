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
