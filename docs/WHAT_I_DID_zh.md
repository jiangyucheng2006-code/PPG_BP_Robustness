# 目前做了什么，以及在哪里看

## 先说明GitHub和本地电脑的区别

GitHub中保存的是代码、配置和说明，不会上传约20 GB的公开数据、训练权重和实验结果。
这些大文件保存在本机D盘，因此只打开GitHub时看不到它们是正常的。

## 已完成的工作

1. 下载并核验PulseDB公开数据。
2. 从官方数据中只提取PPG信号及对应的SBP、DBP标签。
3. 保留官方Calibration-Free测试集，并按受试者划分训练集和验证集。
4. 实现PPG-only XResNet50和XResNet101基础神经网络。
5. 实现训练、验证、测试、模型保存和MAE/RMSE等指标计算。
6. 使用真实数据完成小型训练，确认整条程序可以运行。
7. 使用完整数据完成一次XResNet101显卡前向和反向传播检查。
8. 代码自动测试全部通过并已上传GitHub。

## 下载的数据在哪里

数据总目录：

`D:\Datasets\PPG_BP_Robustness`

主要文件：

- `PulseDB\pulsedb_supplementary_kaggle_v4.zip`：下载的公开压缩包，约17.3 GB。
- `PulseDB\VitalDB_Subsets`：解压后的五个官方VitalDB子集，约17.8 GB。
- `PulseDB\raw\Segment_Files\PulseDB_MIMIC`：另外下载的12名MIMIC受试者原始文件。
- `processed\pulsedb_full`：已经转成模型可以快速读取的数据，约2.44 GB。

完整转换后共有523,080个10秒PPG片段：

- 训练集：412,920个；
- 验证集：52,560个；
- 官方独立测试集：57,600个。

## 代码在哪里

项目目录：

`C:\Users\Lenovo\Desktop\PPG_BP_Robustness`

常用内容：

- `src\ppg_bp\models\xresnet1d.py`：神经网络主体。
- `scripts\prepare_pulsedb_subsets.py`：把官方大数据转换成PPG训练数据。
- `scripts\train_baseline.py`：训练并评估模型。
- `configs\pulsedb_xresnet101.yaml`：正式基础模型配置。
- `outputs`：本机实验结果和模型权重。

## 已有结果在哪里

- `outputs\p000160_audit\first_segment.png`：一段真实PPG和ABP波形图。
- `outputs\p000160_audit\summary.json`：该受试者的数据检查结果。
- `outputs\smoke_xresnet50\metrics.json`：小型训练的指标。
- `outputs\smoke_xresnet50\best.pt`：小型训练保存的模型。

小型训练只用了12名受试者和169个片段，用途是证明程序可以运行，不能作为论文结果。

## 最简单的查看和运行方法

双击项目根目录中的：

`PPG_BP项目入口.bat`

在菜单里可以：

1. 查看目前下载了什么、数据数量和显卡状态；
2. 打开真实波形图；
3. 重新运行一次小型训练演示；
4. 打开结果文件夹；
5. 打开GitHub项目。

## 正式训练

正式训练使用完整数据和XResNet101，预计需要较长时间。命令为：

```powershell
$env:PPG_BP_DATA_ROOT = "D:/Datasets/PPG_BP_Robustness"
python scripts/train_baseline.py `
  --config configs/pulsedb_xresnet101.yaml `
  --output outputs/pulsedb_xresnet101_full
```

正式结果完成后会出现在：

`outputs\pulsedb_xresnet101_full\metrics.json`

