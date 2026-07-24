"""共享测试 fixtures。"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image


@pytest.fixture
def tmp_images(tmp_path):
    """创建临时测试图片用于比对。

    返回 (company_dir, frame_dir, company_image_path, frame_path)
    """
    company_dir = tmp_path / "company"
    frame_dir = tmp_path / "frames"
    company_dir.mkdir()
    frame_dir.mkdir()

    # 创建一张红色产品图（模拟公司图）
    arr1 = np.zeros((200, 200, 3), dtype=np.uint8)
    arr1[:, :, 2] = 200  # 红色
    arr1[50:150, 50:150] = [255, 100, 50]  # 中心橙色块
    img1 = Image.fromarray(arr1)
    company_path = company_dir / "product_red.png"
    img1.save(str(company_path))

    # 创建一张相似的帧图（模拟视频帧）
    arr2 = np.zeros((300, 400, 3), dtype=np.uint8)
    arr2[:, :, 2] = 190  # 略有不同的红色
    arr2[80:180, 100:250] = [250, 110, 60]  # 略偏移的橙色块
    img2 = Image.fromarray(arr2)
    frame_path = frame_dir / "frame_0001.jpg"
    img2.save(str(frame_path), quality=85)

    # 创建一张完全不同的帧图（蓝色）
    arr3 = np.zeros((300, 400, 3), dtype=np.uint8)
    arr3[:, :, 0] = 200  # 蓝色
    img3 = Image.fromarray(arr3)
    diff_frame_path = frame_dir / "frame_0002.jpg"
    img3.save(str(diff_frame_path), quality=85)

    return company_dir, frame_dir, company_path, frame_path, diff_frame_path
