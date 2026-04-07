from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
import io
import base64
from typing import Dict
class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        # 支持从 config 获取设备，或默认 MPS > CPU
        if 'device' in config:
            device = config['device']
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(model_name=config['caption_model_name'], model_name_or_path=config['caption_model_path'], device=device)
        print(f'Omniparser initialized on {device}!')

    def parse(self, image_base64: str):
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print('image size:', image.size)
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        (text, ocr_bbox), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold': 0.8}, use_paddleocr=False)
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD = self.config['BOX_TRESHOLD'], output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

        return dino_labled_img, parsed_content_list

    def parse_from_file(self, image_path: str):
        """
        从文件路径解析图片

        Args:
            image_path: 图片文件路径

        Returns:
            (dino_labled_img_base64, parsed_content_list)
        """
        image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
        return self.parse(image_base64)