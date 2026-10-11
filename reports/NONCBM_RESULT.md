# 非 CBM 对照

使用冻结骨干输出直接训练线性分类头。四组固定学习率／权重衰减组合按验证集 NLL 选择，三个训练种子为 42、43、44。

![原图与换背景准确率](noncbm_background.png)

| 数据与骨干 | 训练视图 | 原图准确率 | 换背景准确率 |
| --- | --- | --- | --- |
| cifar100_cliprn50 | original | 69.36% ± 0.11 | — |
| cifar10_cliprn50 | original | 88.51% ± 0.12 | — |
| cub_rn18_imagenet | original | 61.40% ± 0.20 | 38.67% ± 0.16 |
| cub_rn18_imagenet_paired | original＋changed | 63.08% ± 0.06 | 50.41% ± 0.22 |
| rival10_rn18_imagenet | original | 98.37% ± 0.07 | 89.49% ± 0.13 |
| rival10_rn18_imagenet_paired | original＋changed | 98.35% ± 0.05 | 95.09% ± 0.03 |

CIFAR-10／100 使用既有冻结 CLIP RN50 特征；RIVAL10／CUB 使用本轮共享的 ImageNet ResNet18 特征。误差为独立分类头训练种子间的标准差。
