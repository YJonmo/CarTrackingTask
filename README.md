
This is a demo for car tracking in a road junction using YOLOv5 (https://docs.ultralytics.com). 

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/YJonmo/CarTrackingTask  # clone
cd CarTrackingTask
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Inference</summary>

Bash
```bash
  
python TrackStreet.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/videos/  --device cpu
```
Python
```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

[assets]: https://github.com/ultralytics/yolov5/releases
[tta]: https://github.com/ultralytics/yolov5/issues/303
