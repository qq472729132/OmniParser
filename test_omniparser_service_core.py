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