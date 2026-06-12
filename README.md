# CIFAR-10 / CIFAR-100 Optimizer Comparison

这个项目用于在 CIFAR-10 和 CIFAR-100 上比较同一个图像分类模型在三种手写优化器下的表现：

- SGD
- AdamW
- Muon-AdamW

代码没有使用 `torch.optim`。模型使用 `torch.nn` 定义，梯度由 PyTorch autograd 计算，但参数更新逻辑在 `cifar_muon_compare.py` 中手写完成。

Muon-AdamW 的参数分组规则是：

- `param.ndim >= 2` 的参数使用 Muon，例如卷积核和线性层权重。
- `param.ndim < 2` 的参数使用 AdamW，例如 bias、BatchNorm 的 weight 和 bias。

## 1. 先测试代码功能

最开始不要直接跑完整训练。先用假数据验证代码链路：

```bash
python cifar_muon_compare.py \
  --fake-data \
  --datasets cifar10 cifar100 \
  --optimizers sgd adamw muon_adamw \
  --epochs 1 \
  --limit-train-batches 1 \
  --limit-test-batches 1 \
  --batch-size 8 \
  --width 16 \
  --blocks 1 1 1 1 \
  --num-workers 0 \
  --no-save-model
```

多 seed 功能的最小测试可以再加 `--seeds 42 123`。

这个命令只检查：

- CIFAR-10 和 CIFAR-100 两个分类头都能建立；
- 三种手写优化器都能完成 forward、backward、step；
- loss、accuracy、训练时间、峰值显存能记录；
- `config.json`、`metrics.csv`、`summary.csv`、`summary_mean_std.csv` 和图表能输出。

假数据准确率没有实验意义，只看程序是否报错。

## 2. 小规模真实数据测试

功能测试通过后，先跑少量真实 CIFAR-10 batch。第一次运行会下载数据集到 `./data`。

```bash
python cifar_muon_compare.py \
  --datasets cifar10 \
  --optimizers sgd adamw muon_adamw \
  --epochs 1 \
  --limit-train-batches 10 \
  --limit-test-batches 5 \
  --batch-size 64 \
  --width 32 \
  --blocks 1 1 1 1 \
  --amp \
  --cosine-lr \
  --no-save-model
```

如果这一步正常，再进入正式实验。

## 3. 8GB 显卡正式实验建议

你的显卡是 8GB，所以建议优先保证完整实验能跑完，而不是一开始把模型宽度拉满。

推荐先从下面这组单 seed 正式实验开始：

```bash
python cifar_muon_compare.py \
  --datasets cifar10 cifar100 \
  --optimizers sgd adamw muon_adamw \
  --epochs 20 \
  --batch-size 64 \
  --width 64 \
  --blocks 3 3 3 3 \
  --amp \
  --cosine-lr \
  --warmup-epochs 2 \
  --grad-clip 1.0 \
  --output-dir outputs_width64_e20
```

`width=64, blocks=3 3 3 3` 大约是 1700 万参数级别，已经明显大于原来的简单 CNN，适合 8GB 显卡做正式对比。

如果你要更严谨地报告均值和标准差，可以跑 3 个随机种子。这个会把总训练量变成 3 倍，建议确认单 seed 能稳定跑完后再开：

```bash
python cifar_muon_compare.py \
  --datasets cifar10 cifar100 \
  --optimizers sgd adamw muon_adamw \
  --seeds 42 123 2026 \
  --epochs 20 \
  --batch-size 64 \
  --width 64 \
  --blocks 3 3 3 3 \
  --amp \
  --cosine-lr \
  --warmup-epochs 2 \
  --grad-clip 1.0 \
  --output-dir outputs_width64_e20_3seeds
```

如果显存还有余量，可以尝试更大的模型：

```bash
python cifar_muon_compare.py \
  --datasets cifar10 cifar100 \
  --optimizers sgd adamw muon_adamw \
  --epochs 20 \
  --batch-size 48 \
  --width 80 \
  --blocks 3 3 3 3 \
  --amp \
  --cosine-lr \
  --warmup-epochs 2 \
  --grad-clip 1.0 \
  --output-dir outputs_width80_e20
```

`width=80` 大约是 2700 万参数级别。8GB 显存不建议一开始使用 `width=96` 或 `width=128` 同时跑 CIFAR-10/CIFAR-100 三个优化器。它们可能能跑，但训练时间和显存风险都会明显增加。

## 4. 显存不够时怎么降档

如果出现 CUDA out of memory，按这个顺序降：

1. `--batch-size 64` 改成 `--batch-size 32`。
2. `--width 80` 改成 `--width 64`。
3. `--blocks 3 3 3 3` 改成 `--blocks 2 2 2 2`。
4. 保留 `--amp`，不要关掉。

CIFAR-10 和 CIFAR-100 图片尺寸相同，显存差距主要不来自数据集，而来自模型宽度、batch size、优化器状态和中间激活。

## 5. 输出文件

默认输出到 `./outputs`，也可以用 `--output-dir` 指定目录。

- `config.json`：保存本次命令参数、seed、设备、PyTorch/torchvision 版本、模型参数量。
- `metrics.csv`：逐 epoch 指标，包括 seed、train loss、train acc、test loss、test acc、epoch time、elapsed time、peak memory、skipped batches。
- `summary.csv`：每个 seed / dataset / optimizer 的最终汇总，包括最佳测试准确率、最佳 epoch、总时间、平均 epoch 时间、峰值显存。
- `summary_mean_std.csv`：跨 seed 的均值和标准差；单 seed 时标准差为 0。
- `cifar10_optimizer_comparison_detailed.png`：CIFAR-10 详细图表。
- `cifar100_optimizer_comparison_detailed.png`：CIFAR-100 详细图表。
- `seed*_best.pt`：每组实验测试准确率最高 epoch 的模型权重，除非使用 `--no-save-model`。
- `seed*_last.pt`：每组实验最后一个 epoch 的模型权重，除非使用 `--no-save-model`。

详细图表包含：

- Train Loss 曲线；
- Test Loss 曲线；
- Train/Test Accuracy 曲线；
- 每个 epoch 的训练时间；
- CUDA 峰值显存；
- 最佳测试准确率柱状图。

## 6. 常用参数说明

- `--seed 42`：单随机种子，保持旧用法。
- `--seeds 42 123 2026`：多随机种子重复实验，并输出跨 seed 均值/标准差。
- `--fake-data`：使用假数据，只用于测试代码功能。
- `--aug auto/simple/strong`：`auto` 会在假数据时使用简单增强，真实数据时使用强增强。
- `--amp`：使用 CUDA bfloat16 autocast，建议 8GB 显卡正式实验开启。
- `--cuda-warmup-batches`：正式计时前预热 CUDA，默认 1，减少第一个优化器吃亏。
- `--grad-clip 1.0`：限制梯度范数，Muon 学习率较敏感时更稳。
- `--fail-on-nan`：调试时使用，遇到 NaN/Inf 立即报错。
- `--limit-train-batches` / `--limit-test-batches`：只跑部分 batch，用于快速测试。

## 7. 严谨性建议

建议正式报告采用两阶段：

1. 先跑单 seed，确认 8GB 显存、训练时间、loss 曲线都正常。
2. 再跑 `--seeds 42 123 2026`，用 `summary_mean_std.csv` 报告均值和标准差。

三种优化器会从同一 seed 下的同一份初始权重开始训练，因此优化器之间的对比是公平的。不同 seed 会重新初始化模型，用来估计随机波动。

## 8. 结果解读重点

正式报告里建议比较：

- 同 epoch 下谁的 train loss 更低；
- 同 epoch 下谁的 test accuracy 更高；
- 谁更早达到较高准确率；
- 每个优化器的训练时间；
- 每个优化器的峰值显存；
- CIFAR-10 和 CIFAR-100 上差距是否一致。

不要用假数据结果判断优化器优劣；假数据只用来验证代码功能。
