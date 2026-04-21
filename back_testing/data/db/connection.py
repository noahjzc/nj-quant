"""数据库连接管理"""
import os
from configparser import ConfigParser
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

_config = None
_engine = None
_session_factory = None


def get_db_config() -> dict:
    """读取数据库配置"""
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'database.ini'
        parser = ConfigParser()
        parser.read(config_path)
        _config = dict(parser.items('postgresql'))
    return _config


def get_engine():
    """获取数据库引擎（单例）"""
    global _engine
    if _engine is None:
        config = get_db_config()
        db_url = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        _engine = create_engine(db_url, pool_size=5, max_overflow=10)
    return _engine


def get_session():
    """获取数据库会话"""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(bind=engine)
    return scoped_session(_session_factory)


def close_session():
    """关闭会话"""
    global _session_factory
    if _session_factory is not None:
        _session_factory = None