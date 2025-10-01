파이썬 3.10.13 사용했습니다.

# 1) Create env and install
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# 2) Run
python run_pipeline.py \
  --source DEMOFILE.mp4 \
  --yolo yolo12l.pt \
  --pose_weight yolov11l-pose.pt \
  --out result.jsonl \
  --save_vis result.mp4 \
  --draw_parts \
  --show
