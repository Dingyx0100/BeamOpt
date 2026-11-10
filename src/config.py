"""Load configs/agent.yaml with paths resolved relative to the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / 'configs' / 'agent.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    d = cfg['data']
    d['data_dir'] = str((ROOT / d['data_dir']).resolve())
    return cfg
