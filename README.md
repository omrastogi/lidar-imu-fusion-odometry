# LiDAR-IMU Fusion Odometry

## Downloading the Dataset

The dataset is hosted on HuggingFace at [omrastogi/lidar_imu_odometry](https://huggingface.co/datasets/omrastogi/lidar_imu_odometry).

**Option 1 — HuggingFace CLI**
```bash
pip install huggingface_hub
huggingface-cli download omrastogi/lidar_imu_odometry --repo-type dataset --local-dir data/
```

**Option 2 — Python**
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="omrastogi/lidar_imu_odometry",
    repo_type="dataset",
    local_dir="data/",
)
```

After download, `data/raw/` will match the structure below and the loader will work with no extra configuration.

> **License:** KITTI raw data is released under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Non-commercial use only.

---

## Dataset Structure

```
data/raw/
└── 2011_10_03/                        # date-level grouping (one per recording date)
    ├── calib_cam_to_cam.txt           # shared calibration for all drives on this date
    ├── calib_imu_to_velo.txt
    ├── calib_velo_to_cam.txt
    └── 2011_10_03_drive_0027_sync/    # one folder per drive session
        ├── oxts/
        │   ├── data/                  # per-frame IMU/GPS .txt files (~100 Hz)
        │   └── timestamps.txt
        └── velodyne_points/
            ├── data/                  # per-frame LiDAR .bin files (~10 Hz)
            ├── timestamps.txt
            ├── timestamps_start.txt
            └── timestamps_end.txt
```

To add a new drive session, place it alongside `2011_10_03_drive_0027_sync/`.
Calibration files are shared at the date level; add a new date folder for a different recording date.

---

## KITTI Loader (`kitti_loader.py`)

Loads LiDAR scans, IMU measurements, and calibration from the KITTI raw dataset format.

### Classes

#### `KittiRawLoader(drive_dir, calib_dir="auto")`

Main entry point. Loads one drive session.

| Parameter | Description |
|-----------|-------------|
| `drive_dir` | Path to the drive sync folder (e.g. `data/raw/2011_10_03/2011_10_03_drive_0027_sync`) |
| `calib_dir` | Path to calibration folder. Default `"auto"` detects it as the parent of `drive_dir`. Pass `None` to skip calibration. |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_frames` | int | Number of LiDAR frames |
| `n_imu` | int | Number of IMU samples |
| `velo_timestamps` | ndarray | LiDAR timestamps in seconds relative to frame 0 |
| `oxts_timestamps` | ndarray | IMU timestamps in seconds relative to LiDAR frame 0 |
| `calib` | `KittiCalibration` or `None` | Loaded calibration (see below) |

**Methods**

```python
loader.get_lidar_scan(k)         # → ndarray (N, 4)  x, y, z, reflectance for frame k
loader.get_timestamp(k)          # → float  seconds from frame 0
loader.get_imu_between(k, k+1)  # → list of {"gyro": (3,), "accel": (3,), "dt": float}
```

`get_imu_between(k_start, k_end)` returns all IMU measurements whose timestamps fall in
`[t_{k_start}, t_{k_end})`. Each entry's `dt` spans to the next sample; the last entry's
`dt` spans to `t_{k_end}`, so `sum(dt)` equals the inter-frame interval exactly.
The output is directly compatible with `IMUPreintegrator.integrate_between()`.

---

#### `KittiCalibration(calib_dir)`

Parses the three KITTI calibration files and exposes rigid-body transforms.

**Attributes**

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `T_imu_to_velo` | (4, 4) | IMU frame → LiDAR (Velodyne) frame |
| `T_velo_to_imu` | (4, 4) | LiDAR frame → IMU frame (inverse) |
| `T_velo_to_cam0` | (4, 4) | LiDAR frame → unrectified cam0 frame |
| `T_imu_to_cam0` | (4, 4) | IMU frame → cam0 (composed) |
| `cam[0..3]` | dict | Per-camera: `K` (3×3), `D` (5,), `R_rect` (3×3), `P_rect` (3×4) |

**Coordinate frames note:** raw sensor data is not pre-transformed. LiDAR scans are in
the Velodyne frame; IMU measurements are in the body frame. Apply the calibration
transforms yourself when fusing modalities.

```
IMU frame ──T_imu_to_velo──▶ LiDAR frame ──T_velo_to_cam0──▶ cam0 frame
```

---

### Usage

```python
from kitti_loader import KittiRawLoader

# calib auto-detected from parent directory
loader = KittiRawLoader("data/raw/2011_10_03/2011_10_03_drive_0027_sync")

scan = loader.get_lidar_scan(0)          # (N, 4) float32
t    = loader.get_timestamp(0)           # seconds

imu  = loader.get_imu_between(0, 1)     # list of dicts
# → [{"gyro": array([wx,wy,wz]), "accel": array([ax,ay,az]), "dt": 0.0403}]

T_iv = loader.calib.T_imu_to_velo       # (4, 4)
```

**With `IMUPreintegrator`:**

```python
from imu_preintegration import IMUPreintegrator

preint = IMUPreintegrator()
preint.integrate_between(loader.get_imu_between(k, k + 1))
result = preint.get_result()
# result["delta_R"], result["delta_v"], result["delta_p"] — in IMU frame
```

### Smoke test

```bash
python kitti_loader.py data/raw/2011_10_03/2011_10_03_drive_0027_sync
```