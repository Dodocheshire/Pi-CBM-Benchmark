# 泛化与分类头容量

## cifar10/clip_RN50/original/seed42

![NEC 曲线](nec_0.png)

| 方法 | 稠密头原图准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- |
| LF-CBM | 84.06% | 0 |
| LF-Pi | 85.55% | 0 |
| 非 CBM | 85.79% | — |

## cifar10/resnet18/corruption_augmented/seed42

![NEC 曲线](nec_1.png)

| 方法 | 稠密头原图准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- |
| LF-CBM | 85.92% | 0 |
| LF-Pi | 87.56% | 0 |
| 非 CBM | 87.32% | — |

## cifar10/resnet18/original/seed42

![NEC 曲线](nec_2.png)

| 方法 | 稠密头原图准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- |
| LF-CBM | 86.70% | 0 |
| LF-Pi | 88.21% | 0 |
| 非 CBM | 88.43% | — |

## cifar10/resnet50/original/seed42

![NEC 曲线](nec_3.png)

| 方法 | 稠密头原图准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- |
| LF-CBM | 89.60% | 0 |
| LF-Pi | 91.30% | 0 |
| 非 CBM | 91.54% | — |

## cub/clip_RN50/original/seed42

![NEC 曲线](nec_4.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-CBM | 53.73% | 14.10% | 8 |
| LF-Pi | 64.36% | 23.11% | 0 |
| 非 CBM | 70.65% | 33.17% | — |

## cub/resnet18/original/seed42

![NEC 曲线](nec_5.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-CBM | 44.25% | 21.11% | 11 |
| LF-Pi | 54.02% | 30.74% | 7 |
| 非 CBM | 61.40% | 38.67% | — |

## cub/resnet18/paired/seed42

![NEC 曲线](nec_6.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-CBM | 33.40% | 22.13% | 0 |
| LF-Pi | 50.66% | 39.25% | 0 |
| 非 CBM | 63.08% | 50.41% | — |

## cub/resnet50/paired/seed42

![NEC 曲线](nec_7.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-Pi | 60.36% | 47.98% | 0 |
| 非 CBM | 66.51% | 53.51% | — |

## rival10/clip_RN50/original/seed42

![NEC 曲线](nec_8.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-CBM | 96.91% | 86.82% | 0 |
| LF-Pi | 98.77% | 90.67% | 0 |
| 非 CBM | 99.22% | 92.53% | — |

## rival10/clip_RN50/paired/seed42

![NEC 曲线](nec_9.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-CBM | 96.65% | 91.10% | 0 |
| LF-Pi | 98.88% | 95.87% | 0 |
| 非 CBM | 99.18% | 97.10% | — |

## rival10/resnet18/paired/seed42

![NEC 曲线](nec_10.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-CBM | 92.56% | 85.48% | 0 |
| LF-Pi | 97.33% | 93.56% | 0 |
| PiNI 均值 | 97.99% | 94.74% | 0 |
| PiNI 均值＋噪声 | 97.92% | 94.77% | 0 |
| 非 CBM | 98.35% | 95.08% | — |

## waterbirds/clip_RN50/paired/seed42

![NEC 曲线](nec_11.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| LF-Pi | 97.01% | 93.39% | 0 |
| PiNI 均值 | 96.79% | 93.18% | 0 |
| PiNI 均值＋噪声 | 96.72% | 93.06% | 0 |
| 非 CBM | 96.59% | 92.99% | — |

## waterbirds/resnet50/paired/seed42

![NEC 曲线](nec_12.png)

| 方法 | 稠密头原图准确率 | 稠密头换背景准确率 | SAGA 达到步数上限的路径点 |
| --- | --- | --- | --- |
| PiNI 均值 | 96.32% | 94.43% | 0 |
| PiNI 均值＋噪声 | 96.44% | 94.29% | 0 |
| 非 CBM | 96.85% | 93.27% | — |

## cifar10/resnet18/corruption_augmented/seed42 · 官方 CIFAR-C · 固定分类头

![扰动下 NEC 曲线](nec_corruptions_0.png)

## cifar10/resnet18/original/seed42 · 官方 CIFAR-C · 固定分类头

![扰动下 NEC 曲线](nec_corruptions_1.png)

## cifar10/resnet50/original/seed42 · 官方 CIFAR-C · 固定分类头

![扰动下 NEC 曲线](nec_corruptions_2.png)

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
