# Parameter Sharing Exploration and Hetero center triplet loss for VT Re-ID
Pytorch code for "Parameter Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"[(arxiv)](https://arxiv.org/abs/2008.06223).

### Highlights
- We achieve the state-of-the-art performance on two datasets by large margins, which can be a strong VT Re-ID baseline to boost the future research with high quality.
- We explore the parameter sharing problem in the two stream network. To the best of our knowledge, it is the first attempt to analyze the impact of the number of parameters sharing for cross-modality feature learning.
- We propose the hetero-center triplet loss to constrain the distance of different class centers from both the same modality and cross modality.

### Results
Dataset | Rank1  | mAP | mINP
 ---- | ----- | ------  | -----
 RegDB | ~91.05% | ~83.28%  | ~68.84%
 SYSU-MM01  | ~61.68% | ~57.51% | ~39.54%
 

### Usage
Our code extends the pytorch implementation of Cross-Modal-Re-ID-baseline in [Github](https://github.com/mangye16/Cross-Modal-Re-ID-baseline). Please refer to the offical repo for details of data preparation.

### Training
Train a model by
```bash
python train_mine.py --dataset regdb --gpu 1 --pcb on -- share_net 2 --w_center 2
```
**Parameters**: More parameters can be found in the script and code.

**For SYSU-MM dataset**: batch_size=6, num_pos=8, w_center=1.0，local_feat_dim=256, num_strips=6, label_smooth=off, (p=3.0 for gm_pool).

###  Citation

Please kindly cite the following paper in your publications if it helps your research:
```
@article{liu2020parameter,
  title={Parameter sharing exploration and hetero-center triplet loss for visible-thermal person re-identification},
  author={Liu, Haijun and Tan, Xiaoheng and Zhou, Xichuan},
  journal={IEEE Transactions on Multimedia},
  volume={23},
  pages={4414--4425},
  year={2020},
  publisher={IEEE}
}
```
```
@article{liu2021strong,
  title={Strong but simple baseline with dual-granularity triplet loss for visible-thermal person re-identification},
  author={Liu, Haijun and Chai, Yanxia and Tan, Xiaoheng and Li, Dong and Zhou, Xichuan},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={653--657},
  year={2021},
  publisher={IEEE}
}
```

