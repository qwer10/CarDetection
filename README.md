# CarDetection

## Descriptioin
This project will detect cars in one image using the YOLO model. 
Many of the ideas in this notebook are described in the two YOLO papers: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242).

Because the YOLO model is very computationally expensive to train, I will load pre-trained weights to use.

In this project I used the dataset from [drive.ai](https://www.drive.ai/)

Two main work have done：

1. Use object detection on a car detection dataset
2. Deal with bounding boxes

### Model
the architecture as follows:
![image](https://github.com/qwer10/Attention/blob/master/images/attn_mechanism.png)
![image](https://github.com/qwer10/Attention/blob/master/images/attn_model.png)


## Requirements
1. TensorFlow 
2. python 3 or later
3. please import Keras, faker, tqdm, numpy

## Usage
```
python Attention.py
```
## Resualts
