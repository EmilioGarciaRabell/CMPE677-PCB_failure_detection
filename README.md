# CMPE677-PCB_failure_detection


## Setting up the project:

Create and activate the python virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the requirements:
```
python -m pip install -r requirements.txt
```

Install Pytorch with CUDA separately:
```
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
  ```