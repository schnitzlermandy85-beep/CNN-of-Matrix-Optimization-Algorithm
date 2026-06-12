"""
CIFAR-10 / CIFAR-100 optimizer comparison.

The experiment keeps one CIFAR-sized ResNet-like model fixed and compares:
    1. manual SGD
    2. manual AdamW
    3. manual Muon-AdamW

No torch.optim optimizer is used. Gradients are still produced by PyTorch autograd.

Quick correctness check:
    python cifar_muon_compare.py --fake-data --datasets cifar10 --optimizers sgd adamw muon_adamw \
        --epochs 1 --limit-train-batches 2 --limit-test-batches 1 --batch-size 16 --width 32
"""

import argparse
import copy
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFARResNet(nn.Module):
    """
    A moderately larger CIFAR model.

    With width=64 and blocks=(3, 3, 3, 3), it has roughly 11M parameters.
    Use width=32 for fast smoke tests.
    """

    def __init__(self, num_classes, width=64, blocks=(3, 3, 3, 3), dropout=0.05):
        super().__init__()
        self.in_channels = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(width, blocks[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(width * 2, blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(width * 4, blocks[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(width * 8, blocks[3], stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width * 8, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, dropout):
        layers = [BasicBlock(self.in_channels, out_channels, stride=stride, dropout=dropout)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_decay_params(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.endswith(".bias"):
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay


class ManualSGD:
    def __init__(self, params, lr=0.05, momentum=0.9, weight_decay=5e-4):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_count = 0
        self.velocity = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0:
                grad = grad.add(p, alpha=self.weight_decay)
            if self.momentum > 0:
                v = self.velocity.get(p)
                if v is None:
                    v = torch.zeros_like(p)
                    self.velocity[p] = v
                v.mul_(self.momentum).add_(grad)
                grad = v
            p.add_(grad, alpha=-self.lr)


class ManualAdamW:
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=5e-4,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = {}
        self.v = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        for p in self.params:
            if p.grad is None:
                continue
            if self.weight_decay != 0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            grad = p.grad
            m = self.m.get(p)
            v = self.v.get(p)
            if m is None:
                m = torch.zeros_like(p)
                v = torch.zeros_like(p)
                self.m[p] = m
                self.v[p] = v
            m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


def zeropower_via_newtonschulz5(grad, steps=5, eps=1e-7):
    """
    Orthogonalize a 2D gradient matrix with Newton-Schulz iterations.

    This is the core Muon-style update. Convolution kernels are flattened to
    (out_channels, in_channels * kh * kw) before this function is called.
    """
    original_dtype = grad.dtype
    x = grad.float()
    if x.ndim != 2:
        raise ValueError("Muon orthogonalization expects a 2D matrix")
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    else:
        transposed = False

    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


class ManualMuonAdamW:
    """
    Use Muon for matrix-like parameters and AdamW for vector/scalar parameters.

    Matrix-like means param.ndim >= 2, including Conv2d and Linear weights.
    BatchNorm parameters and biases are optimized by AdamW.
    """

    def __init__(
        self,
        named_params,
        lr=0.01,
        adamw_lr=0.001,
        momentum=0.95,
        weight_decay=5e-4,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        ns_steps=5,
    ):
        self.matrix_params = []
        self.other_params = []
        self.matrix_names = []
        self.other_names = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                self.matrix_names.append(name)
                self.matrix_params.append(param)
            else:
                self.other_names.append(name)
                self.other_params.append(param)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.step_count = 0
        self.velocity = {}
        self.adamw = ManualAdamW(
            self.other_params,
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
        )

    def zero_grad(self):
        for p in self.matrix_params:
            p.grad = None
        self.adamw.zero_grad()

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for p in self.matrix_params:
            if p.grad is None:
                continue
            if self.weight_decay != 0:
                p.mul_(1.0 - self.lr * self.weight_decay)

            grad = p.grad
            v = self.velocity.get(p)
            if v is None:
                v = torch.zeros_like(p)
                self.velocity[p] = v
            v.mul_(self.momentum).add_(grad)

            update = v.reshape(v.shape[0], -1)
            update = zeropower_via_newtonschulz5(update, steps=self.ns_steps)
            update = update.reshape_as(p)
            scale = max(1.0, math.sqrt(p.numel() / p.shape[0]))
            p.add_(update, alpha=-self.lr * scale)

        self.adamw.step()


def build_optimizer(name, model, args):
    if name == "sgd":
        return ManualSGD(
            model.parameters(),
            lr=args.sgd_lr,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
        )
    if name == "adamw":
        return ManualAdamW(
            model.parameters(),
            lr=args.adamw_lr,
            betas=(args.adamw_beta1, args.adamw_beta2),
            eps=args.adamw_eps,
            weight_decay=args.weight_decay,
        )
    if name == "muon_adamw":
        return ManualMuonAdamW(
            model.named_parameters(),
            lr=args.muon_lr,
            adamw_lr=args.muon_adamw_lr,
            momentum=args.muon_momentum,
            weight_decay=args.weight_decay,
            adamw_betas=(args.adamw_beta1, args.adamw_beta2),
            adamw_eps=args.adamw_eps,
            ns_steps=args.muon_ns_steps,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def get_transforms(dataset):
    if dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(dataset)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


def load_data(dataset, batch_size, data_dir, num_workers, fake_data=False):
    if dataset == "cifar10":
        dataset_cls = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset == "cifar100":
        dataset_cls = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_transform, test_transform = get_transforms(dataset)
    if fake_data:
        train_set = torchvision.datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=train_transform,
        )
        test_set = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=test_transform,
        )
    else:
        train_set = dataset_cls(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_set = dataset_cls(
            root=data_dir, train=False, download=True, transform=test_transform
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, num_classes


def cosine_lr(base_lr, epoch, epochs, warmup_epochs=0):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, name, lr_scale):
    if name == "sgd":
        optimizer.lr *= lr_scale
    elif name == "adamw":
        optimizer.lr *= lr_scale
    elif name == "muon_adamw":
        optimizer.lr *= lr_scale
        optimizer.adamw.lr *= lr_scale


def reset_optimizer_lr(optimizer, name, args):
    if name == "sgd":
        optimizer.lr = args.sgd_lr
    elif name == "adamw":
        optimizer.lr = args.adamw_lr
    elif name == "muon_adamw":
        optimizer.lr = args.muon_lr
        optimizer.adamw.lr = args.muon_adamw_lr


def current_lrs(optimizer, name):
    if name == "sgd":
        return f"{optimizer.lr:.6g}"
    if name == "adamw":
        return f"{optimizer.lr:.6g}"
    return f"muon={optimizer.lr:.6g}, adamw={optimizer.adamw.lr:.6g}"


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    limit_batches=None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        num_batches += 1

    return total_loss / max(1, num_batches), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, limit_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        num_batches += 1

    return total_loss / max(1, num_batches), 100.0 * correct / max(1, total)


def cuda_memory_mb(device):
    if device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


@dataclass
class RunResult:
    dataset: str
    optimizer: str
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    epoch_time: float
    elapsed_time: float
    peak_memory_mb: float
    lr: str


def run_one_experiment(dataset, optimizer_name, base_state, args, device, output_dir):
    train_loader, test_loader, num_classes = load_data(
        dataset=dataset,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        fake_data=args.fake_data,
    )
    model = CIFARResNet(
        num_classes=num_classes,
        width=args.width,
        blocks=tuple(args.blocks),
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(copy.deepcopy(base_state[num_classes]))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(optimizer_name, model, args)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows = []
    best_acc = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        reset_optimizer_lr(optimizer, optimizer_name, args)
        if args.cosine_lr:
            if optimizer_name == "sgd":
                base_lr = args.sgd_lr
            elif optimizer_name == "adamw":
                base_lr = args.adamw_lr
            else:
                base_lr = args.muon_lr
            lr = cosine_lr(base_lr, epoch, args.epochs, warmup_epochs=args.warmup_epochs)
            set_optimizer_lr(optimizer, optimizer_name, lr / base_lr)

        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            limit_batches=args.limit_train_batches,
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            limit_batches=args.limit_test_batches,
        )
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start
        peak_memory = cuda_memory_mb(device)
        best_acc = max(best_acc, test_acc)

        row = RunResult(
            dataset=dataset,
            optimizer=optimizer_name,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            epoch_time=epoch_time,
            elapsed_time=elapsed_time,
            peak_memory_mb=peak_memory,
            lr=current_lrs(optimizer, optimizer_name),
        )
        rows.append(row)

        print(
            f"{dataset:8s} | {optimizer_name:10s} | epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:6.2f}% | "
            f"test acc {test_acc:6.2f}% | {epoch_time:6.1f}s | peak {peak_memory:7.1f} MB | "
            f"lr {row.lr}"
        )

    ckpt_path = output_dir / f"{dataset}_{optimizer_name}_last.pt"
    if not args.no_save_model:
        torch.save(model.state_dict(), ckpt_path)
    print(f"{dataset} / {optimizer_name}: best test acc = {best_acc:.2f}%")
    return rows


def save_csv(rows, output_dir):
    path = output_dir / "metrics.csv"
    fieldnames = list(RunResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    print(f"Saved metrics: {path}")


def plot_results(rows, output_dir):
    if not rows:
        return
    grouped = {}
    for row in rows:
        grouped.setdefault(row.dataset, {}).setdefault(row.optimizer, []).append(row)

    for dataset, by_optimizer in grouped.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for optimizer_name, opt_rows in by_optimizer.items():
            opt_rows = sorted(opt_rows, key=lambda r: r.epoch)
            epochs = [r.epoch for r in opt_rows]
            axes[0].plot(epochs, [r.train_loss for r in opt_rows], marker="o", label=optimizer_name)
            axes[1].plot(epochs, [r.test_acc for r in opt_rows], marker="o", label=optimizer_name)

        axes[0].set_title(f"{dataset.upper()} Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_title(f"{dataset.upper()} Test Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        path = output_dir / f"{dataset}_optimizer_comparison.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10/100 manual optimizer comparison")
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"], choices=["cifar10", "cifar100"])
    parser.add_argument("--optimizers", nargs="+", default=["sgd", "adamw", "muon_adamw"], choices=["sgd", "adamw", "muon_adamw"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fake-data", action="store_true", help="Use FakeData for quick code-path tests")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--no-save-model", action="store_true")

    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", nargs=4, type=int, default=[3, 3, 3, 3])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--sgd-lr", type=float, default=0.05)
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--adamw-lr", type=float, default=0.001)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--muon-lr", type=float, default=0.01)
    parser.add_argument("--muon-adamw-lr", type=float, default=0.001)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    return parser.parse_args()


def choose_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    return torch.device(name)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    print("=" * 80)
    print("CIFAR-10 / CIFAR-100 optimizer comparison")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Optimizers: {', '.join(args.optimizers)}")
    print(f"Epochs: {args.epochs}, batch size: {args.batch_size}, fake data: {args.fake_data}")
    print(f"Model: CIFARResNet width={args.width}, blocks={tuple(args.blocks)}")

    base_state = {}
    for num_classes in sorted({10 if d == "cifar10" else 100 for d in args.datasets}):
        set_seed(args.seed)
        model = CIFARResNet(
            num_classes=num_classes,
            width=args.width,
            blocks=tuple(args.blocks),
            dropout=args.dropout,
        )
        base_state[num_classes] = copy.deepcopy(model.state_dict())
        print(f"Initial model for {num_classes} classes: {count_parameters(model):,} trainable params")

    all_rows = []
    for dataset in args.datasets:
        for optimizer_name in args.optimizers:
            set_seed(args.seed)
            rows = run_one_experiment(
                dataset=dataset,
                optimizer_name=optimizer_name,
                base_state=base_state,
                args=args,
                device=device,
                output_dir=output_dir,
            )
            all_rows.extend(rows)

    save_csv(all_rows, output_dir)
    plot_results(all_rows, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
