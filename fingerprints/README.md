# Fingerprints Cache

此目录存放预计算的图片指纹缓存文件，由程序自动生成。

## 文件说明

| 文件 | 用途 |
|------|------|
| `clip_company.pkl` | 公司图库 CLIP embedding 缓存 |
| `orb_company.pkl` | 公司图库 ORB 特征点缓存 |

## 重建缓存

当 `company_images/` 或 `crop_images/` 目录中的图片发生变更后，删除此目录下的 `.pkl` 文件，下次运行 `main.py` 时自动重建。

```bash
# 删除缓存
rm fingerprints/*.pkl

# 重新运行，自动重建
python main.py --url https://www.douyin.com/video/xxx
```

> `.pkl` 文件已在 `.gitignore` 中排除，不会提交到 Git。
