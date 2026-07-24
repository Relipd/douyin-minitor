# 抖音版权图片监测

基于 Playwright + CLIP + ORB 的抖音视频版权图片自动监测工具。支持直链输入、三级漏斗初筛、AI 可选二次细筛，自动生成证据报告。

## 快速开始

### 环境要求

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行环境 |
| Node.js | 18+ | Playwright 依赖 |
| Chrome/Chromium | 最新 | 浏览器自动化 |

### 安装

```bash
# 1. 安装核心依赖
pip install -r requirements.txt

# 2. 安装 Playwright 浏览器
playwright install chromium

# 3. 可选依赖（按需安装）
pip install -r requirements-optional.txt   # YOLO / CLIP / OCR
pip install -r requirements-dev.txt        # 测试
```

### 配置

```bash
# 复制配置模板并填入实际值
cp config.example.yaml config.yaml
```

将需要保护的版权图片放入 `company_images/` 目录。支持 jpg / jpeg / png / bmp / webp 格式。

### 运行

```bash
# 单视频直链
python main.py --url https://www.douyin.com/video/xxx

# 多视频
python main.py --url url1 url2 url3

# 从文件读取链接
python main.py --url-file urls.txt

# 从 Excel B 列读取链接
python main.py --excel 链接.xlsx

# 无头模式
python main.py --headless

# AI 二次细筛（需先有初筛结果）
python main.py --ai-only
```

**首次运行：** 如使用搜索模式需登录抖音，运行 `python login.py` 手动扫码，登录状态保存到 `browser_data/`。

---

## 工作流程

```
┌──────────────────────────────────────────────────────┐
│  ① 视频抓帧                                           │
│  Playwright 打开视频页 → JS 播放 → 定时截图             │
├──────────────────────────────────────────────────────┤
│  ② 三级漏斗初筛 (Phase 1)                              │
│  Layer 1: 垃圾帧过滤（黑屏/白屏/模糊/低熵）             │
│  Layer 2: SSIM 帧间去重                               │
│  Layer 3: CLIP 语义召回 → ORB 特征验证 → 综合评分       │
├──────────────────────────────────────────────────────┤
│  ③ AI 二次细筛 (Phase 2, 可选)                        │
│  多模态 LLM 审核配对对比图 → 判定侵权/疑似/排除         │
├──────────────────────────────────────────────────────┤
│  ④ 证据输出                                           │
│  对比图 + HTML 报告 + JSON → review/{video_id}/        │
└──────────────────────────────────────────────────────┘
```

---

## 匹配原理

### 三级漏斗

| 层级 | 方法 | 作用 |
|------|------|------|
| **Layer 1** | 亮度/熵/清晰度检测 | 过滤黑屏、白屏、模糊等无效帧 |
| **Layer 2** | SSIM 结构相似性 | 相邻帧去重，减少重复计算 |
| **Layer 3a** | CLIP 语义召回 | 高维语义空间匹配，召回 Top-5 候选 |
| **Layer 3b** | ORB 特征验证 | 像素级特征点匹配 (Lowe's ratio test) |

### 综合评分

```
final_score = CLIP_similarity × 40 + min(good_matches × 8, 100) × 0.6
```

| 评分区间 | 含义 |
|---------|------|
| 80-100 | 高度疑似侵权 |
| 50-79 | 需要人工确认 |
| 0-49 | 可能性较低 |

---

## 配置参考

完整参数见 `config.example.yaml`。核心可调参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `capture.num_frames` | 200 | 每视频最大帧数 |
| `capture.interval_seconds` | 0.5 | 帧间间隔 |
| `clip.threshold` | 0.25 | CLIP 召回阈值（提高=减少误报） |
| `clip.top_n` | 5 | 每帧候选数 |
| `orb.min_good_matches` | 6 | ORB 最少匹配点数 |
| `orb.ratio_thresh` | 0.75 | Lowe 比率测试阈值 |

### 阈值调优

| 场景 | 调整方向 |
|------|---------|
| 误报太多 | 增大 `clip.threshold`、增大 `orb.min_good_matches` |
| 漏报太多 | 减小 `orb.min_good_matches`、增大 `capture.num_frames` |
| 长视频 | 增大 `num_frames`，减小 `interval_seconds` |

---

## 目录结构

```
douyin-monitor/
├── main.py                     # 主入口 (Phase 1 + Phase 2)
├── login.py                    # 手动登录工具
├── config.example.yaml         # 配置模板
├── src/
│   ├── crawler.py              # Playwright 浏览器自动化
│   ├── phase1.py               # 三级漏斗管线
│   ├── frame_filter.py         # 垃圾帧过滤
│   ├── deduplicator.py         # SSIM 帧间去重
│   ├── clip_matcher.py         # CLIP 语义匹配
│   ├── orb_verifier.py         # ORB 特征验证
│   ├── ai_screener.py          # AI 多模态二次细筛
│   ├── pair_generator.py       # 对比图生成
│   └── utils.py                # 工具函数
├── tests/                      # 测试
├── company_images/              # 版权图片库（本地维护，不上传）
├── fingerprints/                # 预计算缓存
├── browser_data/                # 浏览器登录态
├── review/                      # 证据报告输出
└── temp/                        # 临时文件
```

---

## 常见问题

### 遇到验证码？

终端会提示"检测到人机验证"，在浏览器中手动完成验证后按 Enter 继续。

### 首次运行需要登录？

运行 `python login.py`，在浏览器中扫码登录抖音，登录状态自动保存。

### 如何更新图库？

新增或删除 `company_images/` 中的图片后，删除 `fingerprints/` 目录下的缓存文件，下次运行自动重建。

### 支持直播监控吗？

当前不支持，直播需要 RTMP 流截图等不同技术方案。

---

## 依赖

```
playwright>=1.40
pyyaml>=6.0
opencv-python>=4.8
imagehash>=4.3
Pillow>=10.0
numpy>=1.24
httpx>=0.24
scikit-image>=0.21
open-clip-torch>=2.24
torch>=2.0
torchvision>=0.15
```

## License

Apache 2.0
