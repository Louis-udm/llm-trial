import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    PYTHONPATH = os.environ.get("PYTHONPATH", "")
    TRANSFORMERS_OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE",0)

config = Config()
