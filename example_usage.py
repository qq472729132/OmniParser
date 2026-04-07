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
