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
        if config is None:
            config = {}

        self.config = config

        # 确定设备 - MPS > CPU
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
            'device': str(self.device),  # 传递设备字符串给Omniparser
        }

        self.parser = Omniparser(parser_config)

        # 将模型移到目标设备 (如果与Omniparser内部使用的不同)
        if str(self.device) != 'cpu':
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