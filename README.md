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

结果位于 [同条件评测](reports/EXPERIMENT_REPORT.md)、[三个训练种子汇总](reports/SEED_RESULT.md)、[多种子 NEC 曲线](reports/NEC_SEED_RESULT.md)、[非 CBM 对照](reports/NONCBM_RESULT.md)、[泛化／NEC 曲线](reports/GENERALIZATION_RESULT.md)、[CIFAR 与官方 CIFAR-C](reports/CIFAR_RESULT.md) 和 [大规模数据集](reports/SCALE_RESULT.md)。数据、模型和逐图预测写入忽略的 `artifacts/`。

## 已执行与继续运行

RIVAL10、CUB、CUB–Places Waterbirds 在 ImageNet ResNet18、ImageNet ResNet50、CLIP RN50 上已完成种子 42 的 LF-CBM／LF-Pi 原图与背景增强训练，以及 PiNI 均值／均值加噪训练。非 CBM 对照在相同三种骨干与两种训练视图上使用 42、43、44 三个分类头训练种子。PiNI 当前训练入口固定种子 42。

ResNet18 的三个数据集已完成前景／背景分离与四类图像腐蚀测试，LF-Pi 和 PiNI 采用单次噪声采样比较。共同分类头 NEC 扫描继续运行，图表按完成结果更新。`configs/matrix.yaml` 记录完整实验计划；完成状态以各组 `report.json` 和逐图预测为准。

本机使用环境 `/home/pengcheng/miniconda3/envs/pi-cbm/bin/python`，已有数据与区域缓存来自 `/home/pengcheng/Pi-CBM/PiNI/artifacts/`。

CIFAR10／100 已完成三种共享骨干的原图训练，以及 ResNet18 的高斯噪声增强训练。8 组 LF-CBM／LF-Pi 条件均完成四类官方 CIFAR-C、各三个严重度的测试，非 CBM 每组使用三个种子。Places365 已完成 ResNet18 的 LF-CBM／LF-Pi 与三个种子的非 CBM 对照，使用 73000／7300／36500 张训练／验证／测试图像。ImageNet 已完成 LF-CBM／LF-Pi 与三个种子的非 CBM 分类头，使用 200000／10000／50000 张训练／验证／测试图像；大规模 NEC 扫描和 PiNI 的 CIFAR 区域标注与训练继续运行。

PiNI 的 CIFAR 入口为 `python -m benchmark.pini_cifar --datasets cifar10 cifar100 --dino-batch 4`。它复用冻结 ResNet18 和同一图像划分，以训练集 CLIP 相似度生成概念监督，调用子模块原有的均值／加噪训练。区域缓存支持中断续跑；只准备原图区域可使用 `--prepare-only --region-views original`。CIFAR 没有人工概念标注，报告使用 CLIP 教师相关性；区域缓存和初始化完成后，还需完成正式训练及预测导出才计入结果。

RIVAL10 的任务预训练 ResNet18 使用 18987 张训练图更新权重、2111 张验证图选轮次，5286 张测试图用于最终评测。共享骨干接口为 `benchmark.task_backbones`；`python -m benchmark.task_backbones audit` 核验权重、特征和像素梯度，`python -m benchmark.task_backbones encode --device cpu --batch 32 --spatial` 生成独立的 `rn18_task_pretrained_train_val_selected` 缓存。核验记录见 [任务骨干来源](reports/task_backbone_audit.json) 和 [CPU 特征核验](reports/rival_task_backbone_cpu_audit.json)。

缓存完成后，`python -m benchmark.task_backbone_queue` 核验数据并运行 LF-CBM／LF-Pi 和三个种子的非 CBM 对照。PiNI 的同骨干适配入口为 `python -m benchmark.pini_task_backbone prepare`，复用共享空间特征和区域缓存；`audit` 检查权重与像素梯度，`train` 调用原有初始化、均值与加噪训练。CPU 核验见 [PiNI 任务骨干适配](reports/pini_task_backbone_cpu_audit.json)。

RIVAL10 任务骨干的两组 LF-CBM／LF-Pi、六个非 CBM 分类头，以及 PiNI 均值／均值加噪训练已完成。背景增强组使用相同的 5282 张有效测试图和 13 个共同概念列展示五种方法；六组 NEC 扫描已完成，曲线按分类头实际连接数绘制。完整共享缓存核验见 [缓存一致性记录](reports/rival_task_cache_audit.json)，分类头核验见 [实际 NEC 记录](reports/rival_task_nec_audit.json)。

CUB 任务骨干从 ImageNet ResNet18 初始化，使用 4796／1198／5794 张训练／验证／测试图，已完成固定 15 轮训练，验证集选择第 11 轮，最终测试准确率 72.47%。`python -m benchmark.cub_task_cache` 生成独立的共享特征缓存；`python -m benchmark.cub_task_training_queue` 完成缓存核验后运行 LF-CBM／LF-Pi 与非 CBM 对照。骨干记录见 [训练与选择核验](reports/cub_task_backbone_training_audit.json)。

CUB 的 PiNI 同骨干入口为 `python -m benchmark.pini_cub_task_backbone prepare`，完整共享缓存就绪后核验图像、概念标签和区域顺序；`train` 调用原有初始化、均值与加噪训练。接口核验见 [PiNI CUB 任务骨干](reports/pini_cub_task_backbone_cpu_audit.json)。

CUB 任务骨干的两视图共享缓存已完成，六份划分文件与原数据的标签、概念和图像编号一致，见 [共享缓存核验](reports/cub_task_cache_audit.json)。六个非 CBM 分类头及 LF-CBM／LF-Pi 的原图、背景增强训练均已完成，见 [预测文件核验](reports/cub_task_predictions_audit.json)。共同分类头扫描继续运行；PiNI 完成了 [共享输入核验](reports/pini_cub_task_cache_audit.json)，等待显存后训练。

`python -m benchmark.lf_replication_queue --workers 2` 正在补跑三种共享骨干、三个有概念标注的数据集、两种训练视图的种子 43／44，共 36 组 LF-CBM／LF-Pi 配对训练。沿用种子 42 的骨干、教师缓存和训练设置；每组完成后导出单次噪声推理预测。运行计划见 [多种子重复实验](reports/lf_replication_plan.json)。

RIVAL10／ImageNet ResNet18、ResNet50 的原图与背景增强训练均已完成种子 42、43、44。`python -m benchmark.report_seeds` 汇总三个种子都完成的组：使用相同有效测试图与三个种子共同保留的概念列，绘制均值、样本标准差及各个种子的结果。ResNet18 两组的三种子 NEC 扫描已完成；`python -m benchmark.report_nec_seeds` 按相同目标预算汇总，并以实际 NEC 的均值和波动绘图。

```bash
python -m benchmark.report_matched
python -m benchmark.report_noncbm
python -m benchmark.report_generalization
python -m benchmark.report_cifar
python -m benchmark.report_scale
python -m benchmark.report_seeds
python -m benchmark.report_nec_seeds
python -m benchmark.head_queue --workers 2
```
