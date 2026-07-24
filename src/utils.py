"""共享工具函数。"""
import os
import re
from pathlib import Path


def resolve_env_vars(obj):
    """递归解析配置中的 ${VAR} 环境变量引用。

    用法:
        config.yaml 中:  api_key: ${AI_API_KEY}
        环境变量:        AI_API_KEY=sk-xxx

    支持嵌套 dict 和 list。
    """
    if isinstance(obj, str):
        def _replace(m):
            return os.environ.get(m.group(1), m.group(0))
        return re.sub(r'\$\{(\w+)\}', _replace, obj)
    elif isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


def load_config(path=None, base_dir=None):
    """加载 YAML 配置并解析环境变量。

    Args:
        path: 配置文件路径。None 时使用 base_dir/config.yaml。
        base_dir: 项目根目录。None 时使用 src/utils.py 所在项目的根目录。

    Returns:
        解析后的配置字典。
    """
    import yaml

    if path is None:
        if base_dir is None:
            # 从 utils.py (src/utils.py) 向上两级得到项目根
            base_dir = Path(__file__).resolve().parent.parent
        else:
            base_dir = Path(base_dir).resolve()
        path = base_dir / "config.yaml"
    else:
        path = Path(path)
        if base_dir is None:
            base_dir = path.parent

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return resolve_env_vars(cfg)


def resolve_path(cfg, key, base_dir=None, default=None):
    """将配置中的相对路径解析为绝对路径。"""
    val = cfg.get(key, default)
    if val is None:
        return None
    p = Path(val)
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    return str(p if p.is_absolute() else Path(base_dir) / p)
