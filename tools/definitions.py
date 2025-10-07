# Created by aaronkueh at 5/4/24
import os
import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    print('Running as python executable')
    ROOT_DIR = os.path.dirname(sys.executable)
elif __file__:
    print('Running as python script')
    ROOT_DIR = os.path.dirname(__file__)

print('Application path: ', ROOT_DIR)

CONFIG_DIR = Path(os.path.join(ROOT_DIR, 'config'))
UTILS_DIR = Path(os.path.join(ROOT_DIR, 'utils'))
DATA_DIR = Path(os.path.join(ROOT_DIR, 'data'))
DOC_DIR = Path(os.path.join(ROOT_DIR, 'doc'))
LOG_DIR = Path(os.path.join(ROOT_DIR, 'log'))
POLICY_DIR = Path(os.path.join(ROOT_DIR, 'policy'))