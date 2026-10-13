# Pi-CBM-Benchmark

比较 LF-CBM、LF-Pi、共享空间加噪 PiNI，以及不经过概念层的标准分类器。评测覆盖概念预测、充分概念连接下的分类能力、准确率—NEC 曲线和训练／测试分布偏移。

## 固定的方法版本

- `methods/LF-Pi`：LF-CBM 基线与 LF-Pi，跟踪 `main`。
- `methods/PiNI`：共享空间加噪版本，跟踪 `simplified-spatial-noise`。

评测代码位于 `benchmark/`，实验设置位于 `configs/`。两个子模块的模型、噪声生成和损失函数保持原样；数据适配、骨干接口、分类头容量扫描和结果统计在本仓库实现。

```bash
git clone --recurse-submodules https://github.com/Dodocheshire/Pi-CBM-Benchmark.git
```

## 评测设置

完整矩阵见 [configs/matrix.yaml](configs/matrix.yaml)。先运行一个种子检查数据与方法匹配，再执行三个种子。原图训练和背景增强训练分开比较，所有方法共用图像清单与冻结骨干版本。

概念指标分别统计人工逐图标注 mAP／相关系数、类级概念标签和 CLIP 教师相关性。概念映射由训练数据或官方属性名称决定。非 CBM 对照在同一图像特征上直接训练分类头。

NEC 网格根据数据集与概念库规模设置，同时保留 NEC=5 和 5–30 的参考区间。参考 [VLG-CBM §5.1 与附录 F](https://arxiv.org/html/2408.01432v3)：使用 GLM-SAGA 正则化路径，选取 NEC 接近目标的权重，再剪除最小连接达到规定平均 NEC。每个点保存实际 NEC、每类连接数与原始路径位置。充分 NEC 对照使用验证集选择的全路径模型和稠密分类头。

结果位于 [同条件评测](reports/EXPERIMENT_REPORT.md)、[三个训练种子汇总](reports/SEED_RESULT.md)、[多种子 NEC 曲线](reports/NEC_SEED_RESULT.md)、[非 CBM 对照](reports/NONCBM_RESULT.md)、[泛化／NEC 曲线](reports/GENERALIZATION_RESULT.md)、[扰动下的概念预测与激活变化](reports/OOD_CONCEPT_RESULT.md)、[扰动 NEC 曲线](reports/OOD_NEC_RESULT.md)、[LF-Pi 注入位置比较](reports/LF_SITE_RESULT.md)、[CIFAR 与官方 CIFAR-C](reports/CIFAR_RESULT.md)、[官方 CIFAR-C NEC 曲线](reports/CIFAR_NEC_RESULT.md) 和 [大规模数据集](reports/SCALE_RESULT.md)。数据、模型和逐图预测写入忽略的 `artifacts/`。

`python -m benchmark.ood_full_queue` 在显存连续三次达到 16 GiB 后，串行导出三个有原生概念标注数据集的完整 PiNI 概念值，用于固定分类头的扰动 NEC 评测。每组使用同一批 256 张测试图和 13 个原图／腐蚀设置；输出位于独立的 `artifacts/ood_fullraw/`。新增保存逻辑核验见 [完整概念导出入口](reports/pini_ood_full_export_cpu_audit.json)。

## 已完成的主要评测

| 数据集 | 骨干 | 方法与训练条件 |
| --- | --- | --- |
| RIVAL10、CUB、CUB–Places Waterbirds | ImageNet ResNet18、ResNet50、CLIP RN50 | LF-CBM／LF-Pi 原图与背景增强训练，PiNI 均值／均值加噪，非 CBM 对照 |
| CIFAR-10、CIFAR-100 | ImageNet ResNet18、ResNet50、CLIP RN50 | 三种骨干的 LF-CBM／LF-Pi 原图训练；ResNet18 高斯噪声增强训练与 PiNI；非 CBM 对照 |
| ImageNet 每类 200 张训练图、Places365 | ImageNet ResNet18 | LF-CBM／LF-Pi、非 CBM、稀疏与稠密分类头 |

三个原生概念数据集的共享骨干 LF-CBM／LF-Pi 使用 42、43、44 三个训练种子；PiNI 使用原入口固定的种子 42。非 CBM 的三个种子对应冻结特征上的分类头训练。CUB 另包含仅在官方训练划分训练的 ResNet18 对照。

泛化评测包括换背景、主体／背景分离，以及高斯噪声、高斯模糊、对比度和 JPEG 压缩。官方 CIFAR-C 每个设置使用全部 10000 张测试图；三个原生概念数据集的腐蚀评测使用固定 256 张图。各方法的原图与扰动测试使用同一批图像编号。

RIVAL10 的 LF-Pi 图像层、内部层、全局特征层比较已完成。图像层与内部层各包含原图训练和背景增强训练，结果见 [注入位置比较](reports/LF_SITE_RESULT.md)。

本轮继续补齐宽骨干的 CIFAR 噪声增强对照与剩余共同分类头。报告只纳入完成并通过核验的结果；逐组状态以 `artifacts/` 中的结果、权重和预测文件为准。

## 运行入口

- 数据与 LF-CBM／LF-Pi：`python -m benchmark.lf_cifar --help`。
- PiNI CIFAR：`python -m benchmark.pini_cifar --help`；宽骨干：`python -m benchmark.pini_cifar_wide --help`。
- 相同概念输出上的稀疏／稠密分类头：`python -m benchmark.head_suite --help`。
- CIFAR 图表：`python -m benchmark.report_cifar`。
- 容量与多种子图表：`python -m benchmark.report_generalization`、`python -m benchmark.report_nec_seeds`。

本机环境为 `/home/pengcheng/miniconda3/envs/pi-cbm/bin/python`。数据、区域缓存、训练权重和逐图预测存放在忽略的 `artifacts/` 中；图表与紧凑统计保存在 `reports/`。
