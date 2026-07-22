# 从这里开始

这个仓库当前已经具备第一版完整基础流程：

1. 从PulseDB官方Google Drive下载受试者文件；
2. 读取每段10秒PPG及其SBP、DBP标签；
3. 转换为适合神经网络快速读取的数组；
4. 使用PPG-only XResNet同时预测SBP和DBP；
5. 输出MAE、RMSE、偏差和误差标准差。

## 当前原则

- 最终推理只输入PPG。
- 同一受试者不能同时出现在训练集和测试集。
- 小样本运行只用于检查程序，不能当作论文结果。
- 正式结果使用PulseDB官方Calibration-Free划分。
- 原始数据、模型权重和实验输出不上传GitHub。

## 本机数据位置

建议统一放在：

`D:\Datasets\PPG_BP_Robustness`

当前真实样本文件位于：

`D:\Datasets\PPG_BP_Robustness\PulseDB\sample\p000160.mat`

## 基础命令

安装项目：

```powershell
python -m pip install -e ".[dev]"
```

检查一名真实受试者：

```powershell
python scripts/audit_pulsedb_subject.py `
  --input D:/Datasets/PPG_BP_Robustness/PulseDB/sample/p000160.mat `
  --output outputs/p000160_audit
```

下载少量真实数据进行调试：

```powershell
python scripts/download_pulsedb_gdrive.py `
  --output D:/Datasets/PPG_BP_Robustness/PulseDB/raw `
  --source mimic --limit 10
```

转换数据：

```powershell
python scripts/prepare_pulsedb.py `
  --input D:/Datasets/PPG_BP_Robustness/PulseDB/raw `
  --output D:/Datasets/PPG_BP_Robustness/processed/smoke
```

正式实验前必须把工程调试划分替换为PulseDB官方Calibration-Free划分。

如果已经下载Kaggle公开的VitalDB完整补充子集，可直接转换官方大文件：

```powershell
python scripts/prepare_pulsedb_subsets.py `
  --train D:/Datasets/PPG_BP_Robustness/PulseDB/VitalDB_Subsets/VitalDB_Train_Subset.mat `
  --test D:/Datasets/PPG_BP_Robustness/PulseDB/VitalDB_Subsets/VitalDB_CalFree_Test_Subset.mat `
  --output D:/Datasets/PPG_BP_Robustness/processed/pulsedb_full
```

该流程保留官方Calibration-Free测试集，并仅从官方训练集按受试者划出验证集。

## 标签范围说明

完整VitalDB子集中约2%至3%的标签超出常用生理范围。默认配置保留全部官方样本，便于与公开基准公平比较。
如果要做异常标签敏感性实验，可在配置文件的`data.label_filter`中设置SBP/DBP上下限；过滤只在读取时生效，不会修改原始数据。
