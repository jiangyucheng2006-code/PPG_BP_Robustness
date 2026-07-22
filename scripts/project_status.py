"""Print a beginner-friendly summary of local project data and results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PPG_BP_DATA_ROOT", r"D:\Datasets\PPG_BP_Robustness"))


def size_text(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.2f} MB"
    return f"{size / 1024:.2f} KB"


def print_file(path: Path) -> None:
    if path.exists():
        print(f"  [已下载] {path.name}: {size_text(path.stat().st_size)}")
    else:
        print(f"  [未找到] {path}")


def main() -> None:
    print("=" * 68)
    print("PPG-BP项目状态")
    print("=" * 68)
    print(f"代码位置：{PROJECT_ROOT}")
    print(f"数据位置：{DATA_ROOT}")

    print("\n1. 已下载的主要公开数据")
    archive = DATA_ROOT / "PulseDB" / "pulsedb_supplementary_kaggle_v4.zip"
    print_file(archive)
    subsets = DATA_ROOT / "PulseDB" / "VitalDB_Subsets"
    if subsets.exists():
        for path in sorted(subsets.glob("*.mat")):
            print_file(path)
    raw_mimic = DATA_ROOT / "PulseDB" / "raw" / "Segment_Files" / "PulseDB_MIMIC"
    print(f"  MIMIC原始受试者文件：{len(list(raw_mimic.glob('*.mat')))}个")

    print("\n2. 已转换、可以直接训练的数据")
    processed = DATA_ROOT / "processed" / "pulsedb_full"
    metadata_path = processed / "dataset_meta.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split = np.load(processed / "split.npy", mmap_mode="r")
        counts = {int(code): int(count) for code, count in zip(*np.unique(split, return_counts=True))}
        print(f"  总片段：{metadata['n_samples']:,}")
        print(f"  训练：{counts.get(0, 0):,}")
        print(f"  验证：{counts.get(1, 0):,}")
        print(f"  官方测试：{counts.get(2, 0):,}")
        print(f"  PPG数组：{size_text((processed / 'ppg.npy').stat().st_size)}")
    else:
        print("  尚未找到完整转换结果。")

    print("\n3. 电脑训练环境")
    print(f"  PyTorch：{torch.__version__}")
    print(f"  CUDA可用：{'是' if torch.cuda.is_available() else '否'}")
    if torch.cuda.is_available():
        print(f"  显卡：{torch.cuda.get_device_name(0)}")

    print("\n4. 已有实验结果")
    metrics_path = PROJECT_ROOT / "outputs" / "smoke_xresnet50" / "metrics.json"
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        print("  小型流程测试：已完成")
        print(f"  使用设备：{result['device']}")
        print(f"  测试SBP MAE：{result['test']['sbp_mae']:.2f} mmHg")
        print(f"  测试DBP MAE：{result['test']['dbp_mae']:.2f} mmHg")
        print("  注意：仅12名受试者、169个片段，不能作为论文结果。")
    else:
        print("  尚未找到小型训练结果。")

    formal_metrics = PROJECT_ROOT / "outputs" / "pulsedb_xresnet101_full" / "metrics.json"
    print(f"  正式完整训练：{'已完成' if formal_metrics.exists() else '尚未运行'}")
    print("\n详细说明：docs\\WHAT_I_DID_zh.md")
    print("=" * 68)


if __name__ == "__main__":
    main()

