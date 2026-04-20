import datetime
import glob
import os

import numpy as np


# KITTI raw dataset loader
#
# Expected on-disk layout:
#   data/raw/<date>/                          <- date folder (calib lives here)
#       calib_cam_to_cam.txt
#       calib_imu_to_velo.txt
#       calib_velo_to_cam.txt
#       <date>_drive_<NNNN>_sync/             <- drive_dir
#           oxts/
#               data/           <- *.txt IMU/GPS files (~100 Hz)
#               timestamps.txt
#           velodyne_points/
#               data/           <- *.bin point clouds  (~10 Hz)
#               timestamps.txt
#               timestamps_start.txt
#               timestamps_end.txt
#
# When calib_dir is not supplied, the loader auto-detects it as the
# parent directory of drive_dir (i.e. the date folder above).
#
# OXTS field indices (0-based):
#   ax=11, ay=12, az=13   (body-frame acceleration, m/s^2)
#   wx=17, wy=18, wz=19   (body-frame angular rate, rad/s)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _parse_timestamps(path):
    """Parse KITTI raw timestamp file -> numpy array of POSIX seconds."""
    ts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: 2011-10-03 12:55:30.177927648  (nanosecond precision)
            date_str, time_str = line.split(" ", 1)
            hms, frac = time_str.split(".", 1)
            us = frac[:6].ljust(6, "0")           # truncate to microseconds
            dt = datetime.datetime.strptime(
                f"{date_str} {hms}.{us}", "%Y-%m-%d %H:%M:%S.%f"
            )
            ts.append(dt.timestamp())
    return np.array(ts)


def _read_kv_file(path):
    """Parse a KITTI calibration key: value file -> dict of str -> ndarray."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            try:
                data[key.strip()] = np.array([float(x) for x in vals.split()])
            except ValueError:
                data[key.strip()] = vals.strip()   # keep as string (e.g. calib_time)
    return data


def _make_T(R_flat, t_flat):
    """Build 4x4 homogeneous transform from flat R (9,) and T (3,)."""
    T = np.eye(4)
    T[:3, :3] = R_flat.reshape(3, 3)
    T[:3,  3] = t_flat
    return T


# ------------------------------------------------------------------
# calibration
# ------------------------------------------------------------------

class KittiCalibration:
    """
    Loads all three KITTI raw calibration files and exposes the key
    transforms and camera intrinsics needed for LiDAR-IMU fusion.

    Attributes
    ----------
    T_imu_to_velo : ndarray (4,4)
        Rigid transform: IMU frame -> LiDAR (Velodyne) frame.
    T_velo_to_imu : ndarray (4,4)
        Inverse of T_imu_to_velo.
    T_velo_to_cam0 : ndarray (4,4)
        Rigid transform: LiDAR frame -> (unrectified) cam0 frame.
    T_imu_to_cam0 : ndarray (4,4)
        Composed: IMU -> cam0.
    cam : dict
        Per-camera calibration keyed by camera index (int 0-3).
        Each value is a dict with:
            K       : (3,3) intrinsic matrix
            D       : (5,)  distortion coefficients
            R_rect  : (3,3) rectification rotation
            P_rect  : (3,4) rectified projection matrix
    """

    def __init__(self, calib_dir):
        self.calib_dir = calib_dir
        self._load_imu_to_velo()
        self._load_velo_to_cam()
        self._load_cam_to_cam()
        self.T_imu_to_cam0 = self.T_velo_to_cam0 @ self.T_imu_to_velo

    def _load_imu_to_velo(self):
        d = _read_kv_file(os.path.join(self.calib_dir, "calib_imu_to_velo.txt"))
        self.T_imu_to_velo = _make_T(d["R"], d["T"])
        self.T_velo_to_imu = np.linalg.inv(self.T_imu_to_velo)

    def _load_velo_to_cam(self):
        d = _read_kv_file(os.path.join(self.calib_dir, "calib_velo_to_cam.txt"))
        self.T_velo_to_cam0 = _make_T(d["R"], d["T"])

    def _load_cam_to_cam(self):
        d = _read_kv_file(os.path.join(self.calib_dir, "calib_cam_to_cam.txt"))
        self.cam = {}
        for idx in range(4):
            tag = "%02d" % idx
            K = d["K_%s" % tag].reshape(3, 3)
            D = d["D_%s" % tag]
            R_rect = d["R_rect_%s" % tag].reshape(3, 3)
            P_rect = d["P_rect_%s" % tag].reshape(3, 4)
            self.cam[idx] = {
                "K":      K,
                "D":      D,
                "R_rect": R_rect,
                "P_rect": P_rect,
            }


# ------------------------------------------------------------------
# data loader
# ------------------------------------------------------------------

class KittiRawLoader:
    """
    Loads LiDAR scans, IMU measurements, and calibration for a single
    KITTI raw drive sync folder.
    """

    def __init__(self, drive_dir, calib_dir="auto"):
        self.drive_dir = os.path.abspath(drive_dir)
        self.velo_dir  = os.path.join(self.drive_dir, "velodyne_points", "data")
        self.oxts_dir  = os.path.join(self.drive_dir, "oxts", "data")

        if not os.path.isdir(self.velo_dir):
            raise FileNotFoundError("velodyne_points/data not found: %s" % self.velo_dir)
        if not os.path.isdir(self.oxts_dir):
            raise FileNotFoundError("oxts/data not found: %s" % self.oxts_dir)

        self.velo_files = sorted(glob.glob(os.path.join(self.velo_dir, "*.bin")))
        self.oxts_files = sorted(glob.glob(os.path.join(self.oxts_dir, "*.txt")))

        self.n_frames = len(self.velo_files)
        self.n_imu    = len(self.oxts_files)

        if self.n_frames == 0:
            raise RuntimeError("no .bin files in %s" % self.velo_dir)

        self._load_velo_timestamps()
        self._load_oxts_timestamps()
        self._load_oxts_data()

        if calib_dir == "auto":
            calib_dir = self._detect_calib_dir()
        self.calib = KittiCalibration(calib_dir) if calib_dir else None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _detect_calib_dir(self):
        parent = os.path.dirname(self.drive_dir)
        if os.path.isfile(os.path.join(parent, "calib_imu_to_velo.txt")):
            return parent
        return None

    def _load_velo_timestamps(self):
        path = os.path.join(self.drive_dir, "velodyne_points", "timestamps.txt")
        abs_ts = _parse_timestamps(path)
        self._velo_ts_abs    = abs_ts
        self.velo_timestamps = abs_ts - abs_ts[0]

    def _load_oxts_timestamps(self):
        path = os.path.join(self.drive_dir, "oxts", "timestamps.txt")
        abs_ts = _parse_timestamps(path)
        self._oxts_ts_abs    = abs_ts
        self.oxts_timestamps = abs_ts - self._velo_ts_abs[0]

    def _load_oxts_data(self):
        self.imu_data = []
        for fpath in self.oxts_files:
            vals = np.fromstring(open(fpath).read(), sep=" ")
            self.imu_data.append({
                "accel": vals[11:14],   # ax, ay, az  (body frame, m/s^2)
                "gyro":  vals[17:20],   # wx, wy, wz  (body frame, rad/s)
            })

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_lidar_scan(self, k):
        assert 0 <= k < self.n_frames, "frame %d out of range [0, %d)" % (k, self.n_frames)
        return np.fromfile(self.velo_files[k], dtype=np.float32).reshape(-1, 4)

    def get_timestamp(self, k):
        assert 0 <= k < self.n_frames
        return float(self.velo_timestamps[k])

    def get_imu_between(self, k_start, k_end):
        """
        Return IMU measurements between LiDAR frames k_start and k_end.

        Each element: {"gyro": ndarray(3,), "accel": ndarray(3,), "dt": float}

        dt for each sample = time to the NEXT sample's timestamp.
        The last sample's dt spans from its timestamp to t1 (the k_end
        LiDAR timestamp) -- NOT the full inter-frame interval.
        """
        assert 0 <= k_start < k_end < self.n_frames

        t0 = self._velo_ts_abs[k_start]
        t1 = self._velo_ts_abs[k_end]

        idxs = [i for i, t in enumerate(self._oxts_ts_abs) if t0 <= t < t1]

        if not idxs:
            return []

        measurements = []
        for n, i in enumerate(idxs):
            if n + 1 < len(idxs):
                # dt to the next IMU sample
                dt = float(self._oxts_ts_abs[idxs[n + 1]] - self._oxts_ts_abs[i])
            else:
                # last sample: dt spans from this sample to the k_end LiDAR frame
                dt = float(t1 - self._oxts_ts_abs[i])
            measurements.append({
                "gyro":  self.imu_data[i]["gyro"].copy(),
                "accel": self.imu_data[i]["accel"].copy(),
                "dt":    dt,
            })

        return measurements


# ------------------------------------------------------------------
# quick smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="KITTI raw dataset loader smoke test")
    parser.add_argument("drive_dir")
    parser.add_argument("--calib-dir", default="auto")
    args = parser.parse_args()

    calib_dir = None if args.calib_dir == "none" else args.calib_dir
    loader = KittiRawLoader(args.drive_dir, calib_dir=calib_dir)

    print("drive:          %s" % args.drive_dir)
    print("LiDAR frames:   %d" % loader.n_frames)
    print("IMU samples:    %d" % loader.n_imu)
    print("duration:       %.1f s" % loader.get_timestamp(loader.n_frames - 1))

    scan = loader.get_lidar_scan(0)
    print("scan 0:         %d points  (x y z reflectance)" % scan.shape[0])

    imu_batch = loader.get_imu_between(0, 1)
    print("IMU meas 0->1:  %d samples, total dt=%.4f s" % (
        len(imu_batch),
        sum(m["dt"] for m in imu_batch),
    ))
    if imu_batch:
        m0 = imu_batch[0]
        print("  first accel:  %s m/s^2" % str(np.round(m0["accel"], 4)))
        print("  first gyro:   %s rad/s" % str(np.round(m0["gyro"],  4)))

    if loader.calib:
        c = loader.calib
        print("\ncalibration:")
        print("  T_imu_to_velo:\n%s" % np.round(c.T_imu_to_velo, 6))
        print("  T_velo_to_cam0:\n%s" % np.round(c.T_velo_to_cam0, 6))
        print("  T_imu_to_cam0:\n%s" % np.round(c.T_imu_to_cam0, 6))