# 泛化与分类头容量

## cub/resnet18/paired/seed42

![NEC 曲线](nec_0.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-CBM | 33.40% | 22.13% | 0 |
| LF-Pi（8次采样） | 50.14% | 39.49% | 0 |

## rival10/resnet18/paired/seed42

![NEC 曲线](nec_1.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| PiNI 均值 | 97.99% | 94.74% | 0 |
| PiNI 均值＋噪声 | 97.92% | 94.77% | 0 |

## CUB 图像扰动

本组采用单次噪声采样。

![扰动泛化](corruptions_cub.png)

![主体与背景干预](masked_cub.png)

## RIVAL10 图像扰动

本组采用单次噪声采样。

![扰动泛化](corruptions_rival10.png)

![主体与背景干预](masked_rival10.png)

## WATERBIRDS 图像扰动

本组采用单次噪声采样。

![扰动泛化](corruptions_waterbirds.png)

![主体与背景干预](masked_waterbirds.png)

![水鸟两类平均准确率](masked_waterbirds_balanced.png)
