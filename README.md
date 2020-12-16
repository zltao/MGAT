# MGAT
This is our Pytorch implementation for our paper- Multimodal Graph Attention Network(MGAT):

>	Zhulin Tao, Yinwei Wei, Xiang Wang, Xiangnan He, Xianglin Huang, Tat-Seng Chua:
MGAT: Multimodal Graph Attention Network for Recommendation. Inf. Process. Manag. 57(5): 102277 (2020)

## Introduction
In this work, we propose a new Multimodal Graph Attention Network, short for MGAT, which disentangles personal interests at the granularity of modality. In particular, built upon multimodal interaction graphs, MGAT conducts information propagation within individual graphs, while leveraging the gated attention mechanism to identify varying importance scores of different modalities to user preference.
## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* torch==1.7.0
* numpy==1.16.1
* torch_geometric==1.6.1

## run
```python
CUDA_VISIBLE_DEVICES=0 python  -u train.py --num_epoch 200 --batch_size 2048 --weight_decay 0.1 --l_r 3e-5
```
# Citation
@article{DBLP:journals/ipm/TaoWWHHC20,
  author    = {Zhulin Tao and
               Yinwei Wei and
               Xiang Wang and
               Xiangnan He and
               Xianglin Huang and
               Tat{-}Seng Chua},
  title     = {{MGAT:} Multimodal Graph Attention Network for Recommendation},
  journal   = {Inf. Process. Manag.},
  volume    = {57},
  number    = {5},
  pages     = {102277},
  year      = {2020}
