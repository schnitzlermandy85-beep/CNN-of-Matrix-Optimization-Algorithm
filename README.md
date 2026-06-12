# CIFAR-10 / CIFAR-100 Optimizer Comparison

这个新实验用于比较同一个 CIFAR 图像分类模型在三种手写优化器下的表现：

- SGD
- AdamW
- Muon-AdamW

代码没有使用 `torch.optim`。模型仍然使用 `torch.nn` 定义，梯度由 autograd 计算，参数更新逻辑在 `cifar_muon_compare.py` 里手写完成。

## 先测试代码功能

最开始建议用假数据和很小模型验证代码链路：

```bash
python cifar_muon_compare.py \
  --fake-data \
  --datasets cifar10 \
  --optimizers sgd adamw muon_adamw \
  --epochs 1 \
  --limit-train-batches 2 \
  --limit-test-batches 1 \
  --batch-size 16 \
  --width 32 \
  --blocks 1 1 1 1 \
  --num-workers 0 \
  --no-save-model
```

这个命令只检查：

- CIFAR-like 数据能进入模型；
- SGD / AdamW / Muon-AdamW 都能完成一次反向传播和参数更新；
- loss、accuracy、训练时间、显存统计能正常记录；
- CSV 和曲线图能输出。

## 小规模真实数据测试

确认功能没问题后，可以先跑 CIFAR-10 的少量 batch：

```bash
python cifar_muon_compare.py \
  --datasets cifar10 \
  --optimizers sgd adamw muon_adamw \
  --epochs 1 \
  --limit-train-batches 10 \
  --limit-test-batches 5 \
  --batch-size 64 \
  --width 32 \
  --blocks 1 1 1 1
```

第一次运行会下载数据集到 `./data`。

## 同时比较 CIFAR-10 和 CIFAR-100

正式一点的短实验：

```bash
python cifar_muon_compare.py \
  --datasets cifar10 cifar100 \
  --optimizers sgd adamw muon_adamw \
  --epochs 5 \
  --batch-size 128 \
  --width 64 \
  --blocks 3 3 3 3 \
  --cosine-lr
```

默认 `width=64, blocks=3 3 3 3` 的模型参数量约千万级，比原来的简单 CNN 更大。你的 5060 显卡跑 CIFAR 尺寸图片一般可以从这个设置开始；如果显存不够，先把 `--batch-size` 降到 `64`，再把 `--width` 降到 `48` 或 `32`。

## 输出文件

默认输出到 `./outputs`：

- `metrics.csv`：每个 dataset / optimizer / epoch 的训练损失、训练准确率、测试损失、测试准确率、耗时、峰值显存。
- `cifar10_optimizer_comparison.png`：CIFAR-10 对比曲线。
- `cifar100_optimizer_comparison.png`：CIFAR-100 对比曲线。
- `*_last.pt`：每组实验最后一个 epoch 的模型权重，除非使用 `--no-save-model`。

## Muon-AdamW 参数分组

`ManualMuonAdamW` 的规则是：

- `param.ndim >= 2` 的参数使用 Muon 更新，例如卷积核和线性层权重。
- `param.ndim < 2` 的参数使用 AdamW 更新，例如 bias、BatchNorm 的 weight 和 bias。

这对应你要的“二维参数用 Muon，其他维度用 AdamW”的对比方式。
