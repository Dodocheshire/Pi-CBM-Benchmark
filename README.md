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

结果位于 [同条件评测](reports/EXPERIMENT_REPORT.md)、[非 CBM 对照](reports/NONCBM_RESULT.md)、[泛化／NEC 曲线](reports/GENERALIZATION_RESULT.md)、[CIFAR 与官方 CIFAR-C](reports/CIFAR_RESULT.md) 和 [大规模数据集](reports/SCALE_RESULT.md)。数据、模型和逐图预测写入忽略的 `artifacts/`。

## 已执行与继续运行

RIVAL10、CUB、CUB–Places Waterbirds 在 ImageNet ResNet18、ImageNet ResNet50、CLIP RN50 上已完成种子 42 的 LF-CBM／LF-Pi 原图与背景增强训练，以及 PiNI 均值／均值加噪训练。非 CBM 对照在相同三种骨干与两种训练视图上使用 42、43、44 三个分类头训练种子。PiNI 当前训练入口固定种子 42。

ResNet18 的三个数据集已完成前景／背景分离与四类图像腐蚀测试，LF-Pi 和 PiNI 采用单次噪声采样比较。共同分类头 NEC 扫描继续运行，图表按完成结果更新。`configs/matrix.yaml` 记录完整实验计划；完成状态以各组 `report.json` 和逐图预测为准。

本机使用环境 `/home/pengcheng/miniconda3/envs/pi-cbm/bin/python`，已有数据与区域缓存来自 `/home/pengcheng/Pi-CBM/PiNI/artifacts/`。

CIFAR10／100 的 ResNet18 原图训练和高斯噪声增强训练，以及四类官方 CIFAR-C、各三个严重度的测试已完成；ResNet50 和 CLIP RN50 继续扩展。Places365 已完成 ResNet18 的 LF-CBM／LF-Pi 与三个种子的非 CBM 对照，使用 73000／7300／36500 张训练／验证／测试图像。ImageNet 1000 类训练和 PiNI 的 CIFAR 区域标注与训练继续运行。

PiNI 的 CIFAR 入口为 `python -m benchmark.pini_cifar --datasets cifar10 cifar100 --dino-batch 4`。它复用冻结 ResNet18 和同一图像划分，以训练集 CLIP 相似度生成概念监督，调用子模块原有的均值／加噪训练。区域缓存支持中断续跑；只准备原图区域可使用 `--prepare-only --region-views original`。CIFAR 没有人工概念标注，报告使用 CLIP 教师相关性；区域缓存和初始化完成后，还需完成正式训练及预测导出才计入结果。

RIVAL10 的任务预训练 ResNet18 使用 18987 张训练图更新权重、2111 张验证图选轮次，5286 张测试图用于最终评测。共享骨干接口为 `benchmark.task_backbones`；`python -m benchmark.task_backbones audit` 核验权重、特征和像素梯度，`python -m benchmark.task_backbones encode --device cpu --batch 32 --spatial` 生成独立的 `rn18_task_pretrained_train_val_selected` 缓存。核验记录见 [任务骨干来源](reports/task_backbone_audit.json) 和 [CPU 特征核验](reports/rival_task_backbone_cpu_audit.json)。

```bash
python -m benchmark.report_matched
python -m benchmark.report_noncbm
python -m benchmark.report_generalization
python -m benchmark.report_cifar
python -m benchmark.report_scale
python -m benchmark.head_queue --workers 2
```
