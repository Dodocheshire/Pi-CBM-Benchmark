# 同条件评测结果

所有图表按骨干、训练视图和训练种子分组。分类比较使用相同测试图像；概念比较使用该组 LF-CBM 保留的共同概念列。

LF-CBM／LF-Pi 使用冻结 CLIP 概念相似度监督；PiNI 使用原方法的属性标签监督。CUB 评测使用逐图人工属性，已知标注按 certainty ≥ 3 筛选。

本轮 Waterbirds 为完整 CUB 鸟类按陆鸟／水鸟分组、配合 Places 换背景的二分类任务。

## CUB · clip_RN50 · 背景增强训练 · 种子 42

共同概念 86／候选 112。

![分类结果](matched_classification_0.png)

![概念结果](matched_concepts_0.png)

![相关性与概念偏移](matched_correlation_shift_0.png)

## CUB · clip_RN50 · 原图训练 · 种子 42

共同概念 109／候选 112。

![分类结果](matched_classification_1.png)

![概念结果](matched_concepts_1.png)

![相关性与概念偏移](matched_correlation_shift_1.png)

## CUB · resnet18 · 背景增强训练 · 种子 42

共同概念 37／候选 112。

![分类结果](matched_classification_2.png)

![概念结果](matched_concepts_2.png)

![相关性与概念偏移](matched_correlation_shift_2.png)

## CUB · resnet18 · 原图训练 · 种子 42

共同概念 50／候选 112。

![分类结果](matched_classification_3.png)

![概念结果](matched_concepts_3.png)

![相关性与概念偏移](matched_correlation_shift_3.png)

## CUB · resnet50 · 背景增强训练 · 种子 42

共同概念 55／候选 112。

![分类结果](matched_classification_4.png)

![概念结果](matched_concepts_4.png)

![相关性与概念偏移](matched_correlation_shift_4.png)

## CUB · resnet50 · 原图训练 · 种子 42

共同概念 95／候选 112。

![分类结果](matched_classification_5.png)

![概念结果](matched_concepts_5.png)

![相关性与概念偏移](matched_correlation_shift_5.png)

## RIVAL10 · clip_RN50 · 背景增强训练 · 种子 42

共同概念 17／候选 18。

![分类结果](matched_classification_6.png)

![概念结果](matched_concepts_6.png)

![相关性与概念偏移](matched_correlation_shift_6.png)

## RIVAL10 · clip_RN50 · 原图训练 · 种子 42

共同概念 16／候选 18。

![分类结果](matched_classification_7.png)

![概念结果](matched_concepts_7.png)

![相关性与概念偏移](matched_correlation_shift_7.png)

## RIVAL10 · resnet18 · 背景增强训练 · 种子 42

共同概念 14／候选 18。

![分类结果](matched_classification_8.png)

![概念结果](matched_concepts_8.png)

![相关性与概念偏移](matched_correlation_shift_8.png)

## RIVAL10 · resnet18 · 原图训练 · 种子 42

共同概念 13／候选 18。

![分类结果](matched_classification_9.png)

![概念结果](matched_concepts_9.png)

![相关性与概念偏移](matched_correlation_shift_9.png)

## RIVAL10 · resnet50 · 背景增强训练 · 种子 42

共同概念 16／候选 18。

![分类结果](matched_classification_10.png)

![概念结果](matched_concepts_10.png)

![相关性与概念偏移](matched_correlation_shift_10.png)

## RIVAL10 · resnet50 · 原图训练 · 种子 42

共同概念 14／候选 18。

![分类结果](matched_classification_11.png)

![概念结果](matched_concepts_11.png)

![相关性与概念偏移](matched_correlation_shift_11.png)

## WATERBIRDS · clip_RN50 · 背景增强训练 · 种子 42

共同概念 86／候选 112。

![分类结果](matched_classification_12.png)

![概念结果](matched_concepts_12.png)

![相关性与概念偏移](matched_correlation_shift_12.png)

## WATERBIRDS · clip_RN50 · 原图训练 · 种子 42

共同概念 109／候选 112。

![分类结果](matched_classification_13.png)

![概念结果](matched_concepts_13.png)

![相关性与概念偏移](matched_correlation_shift_13.png)

## WATERBIRDS · resnet18 · 背景增强训练 · 种子 42

共同概念 37／候选 112。

![分类结果](matched_classification_14.png)

![概念结果](matched_concepts_14.png)

![相关性与概念偏移](matched_correlation_shift_14.png)

## WATERBIRDS · resnet18 · 原图训练 · 种子 42

共同概念 50／候选 112。

![分类结果](matched_classification_15.png)

![概念结果](matched_concepts_15.png)

![相关性与概念偏移](matched_correlation_shift_15.png)

## WATERBIRDS · resnet50 · 背景增强训练 · 种子 42

共同概念 55／候选 112。

![分类结果](matched_classification_16.png)

![概念结果](matched_concepts_16.png)

![相关性与概念偏移](matched_correlation_shift_16.png)

## WATERBIRDS · resnet50 · 原图训练 · 种子 42

共同概念 95／候选 112。

![分类结果](matched_classification_17.png)

![概念结果](matched_concepts_17.png)

![相关性与概念偏移](matched_correlation_shift_17.png)

## CUB · 原图训练 · 骨干间共同概念 49 个

![骨干间共同概念比较](cross_backbone_cub_原图训练.png)

## CUB · 背景增强训练 · 骨干间共同概念 35 个

![骨干间共同概念比较](cross_backbone_cub_背景增强训练.png)

## RIVAL10 · 原图训练 · 骨干间共同概念 13 个

![骨干间共同概念比较](cross_backbone_rival10_原图训练.png)

## RIVAL10 · 背景增强训练 · 骨干间共同概念 14 个

![骨干间共同概念比较](cross_backbone_rival10_背景增强训练.png)

## WATERBIRDS · 原图训练 · 骨干间共同概念 49 个

![骨干间共同概念比较](cross_backbone_waterbirds_原图训练.png)

## WATERBIRDS · 背景增强训练 · 骨干间共同概念 35 个

![骨干间共同概念比较](cross_backbone_waterbirds_背景增强训练.png)
