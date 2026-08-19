"""pytest conftest：将项目根目录及子包加入 sys.path，使测试与生产代码均可无缝导入。"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
_SUBDIRS = [
    ROOT_DIR,
    ROOT_DIR / "storage",
    ROOT_DIR / "connectors",
    ROOT_DIR / "generator",
    ROOT_DIR / "evaluation",
]
for p in _SUBDIRS:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
