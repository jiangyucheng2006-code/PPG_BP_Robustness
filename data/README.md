# Data

PulseDB is distributed by its original authors and is not redistributed in
this repository.

- PulseDB repository: https://github.com/pulselabteam/PulseDB
- PulseDB public files: https://drive.google.com/drive/folders/10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV

Expected layout:

```text
<DATA_ROOT>/
├── PulseDB/
│   ├── raw/
│   └── VitalDB_Subsets/
│       ├── VitalDB_Train_Subset.mat
│       └── VitalDB_CalFree_Test_Subset.mat
└── processed/
    ├── smoke/
    └── pulsedb_full/
```

The subject-hash split produced by `scripts/prepare_pulsedb.py` is restricted
to engineering checks. Reported experiments should use the official
calibration-free test set through `scripts/prepare_pulsedb_subsets.py`.
