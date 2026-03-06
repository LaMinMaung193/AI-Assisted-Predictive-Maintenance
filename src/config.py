#-------------- dataset configs -----------
RANDOM_STATE = 42
TEST_SIZE = 0.3   # 70/30 split



#---------------  model save  -------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

