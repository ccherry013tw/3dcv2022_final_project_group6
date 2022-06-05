# 3dcv2022_final_project_group6
## Face Landmarks
We used the pre-trained model (https://github.com/Jiahao-UTS/SLPT-master.git). 
Please download and install by yourself if needed.
The result images are in images/3dcv_pred.
## How to run:
### Try on Cosmetics:  
```python
python try_lips {image_file_path} {landmarks_file_path} {r} {g} {b}
```
### Try on Accessories:
```python
python3 plot_obj.py {object_name} {img_ID}
```
- object name
  1. glasses_1
  2. glasses_2
  3. earring_1
  4. earring_2
- img_ID 
  - -1 to generate all images

## Environment
- Face Landmarks:
  - Python3.8
  - Pytorch 1.10.2
  - Windows 10

- Try on Cosmetics and Accessories:
  - Python3.8
  - Open3d 0.12.0
  - Opencv-python 4.5.1.48
