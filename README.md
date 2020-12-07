# MGAT
This is our Pytorch implementation for our paper- Multimodal Graph Attention Network(MGAT):

>	Zhulin Tao, Yinwei Wei, Xiang Wang, Xiangnan He, Xianglin Huang, Tat-Seng Chua:
MGAT: Multimodal Graph Attention Network for Recommendation. Inf. Process. Manag. 57(5): 102277 (2020)

## Introduction
In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN, including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* torch==1.7.0
* numpy==1.16.1
* torch_geometric==1.6.1

## run
```python
CUDA_VISIBLE_DEVICES=0 python  -u train.py --num_epoch 200 --batch_size 2048 --weight_decay 0.1 --l_r 3e-5
```
