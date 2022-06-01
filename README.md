

This is a demo for car tracking in a road junction using YOLOv5 (https://docs.ultralytics.com) and view synthesis. 

The instructions for creating this repository is available in the Documentation folder. 

The demo is availabe on google colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a6_CKuhk88YsjBBVS5YiDnZl_Ka4xthQ?usp=sharing)


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

  
Weights:
  
yolov5n.pt is the fastest but least accurate
  
yolov5s.pt is slower than 'n' but more accurate
  
yolov5m.pt is balance between speed and accuracy
  
yolov5l.pt is more accurate than 'm' but slower
  
yolov5x.pt is the most accurate model but slowest
  
  
 Use the following bash command to run the code. You may remove the '--view-img' flag to increase the processing speed.   
  
```bash
  
python track_cars.py --yolo-model yolov5s.pt --deep-sort-model osnet_x0_5_market1501 --conf-thres 0.25 --source data/videos --output-path result --view-img   
  
```
