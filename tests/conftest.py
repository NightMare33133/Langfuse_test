"""pytest conftest：将项目根目录加入 sys.path，使测试可导入生产模块。"""

import sys
from pathlib import Path

# 项目根目录（tests/ 的上级目录）
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
