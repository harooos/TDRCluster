"""
简化的配置管理器 - 纯配置文件模式
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def _load_config() -> Dict[str, Any]:
    """
    加载配置文件
    
    Returns:
        Dict[str, Any]: 配置字典
    """
    config_path = Path(__file__).parent / "config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ 配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return {}


# 模块加载时直接初始化全局CONFIG
CONFIG = _load_config()