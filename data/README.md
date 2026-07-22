# Data

Do not commit raw physiological datasets to GitHub.

The current Windows workstation uses this external data root:

`D:\Datasets\PPG_BP_Robustness`

Recommended layout:

- `PulseDB/raw/`: official MATLAB subject files or official subset archives.
- `PulseDB/sample/`: a few real files used for engineering checks.
- `processed/smoke/`: small model-ready arrays.
- `processed/pulsedb_full/`: full model-ready arrays.

PulseDB v2 official public folder:

https://drive.google.com/drive/folders/10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV

The scripts in `scripts/` provide resumable downloading, conversion, auditing,
and training. The deterministic 80/10/10 split in the converter is only for
engineering checks. Publication experiments must use PulseDB's official
calibration-free subject split.
