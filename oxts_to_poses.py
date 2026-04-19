import numpy as np
import glob
import os
import argparse


# ------------------------------------------------------------------
# constants
# ------------------------------------------------------------------

EARTH_RADIUS = 6378137.0       # WGS84 equatorial radius in meters


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _read_oxts_file(path):
    """Parse a single OXTS text file -> dict of named fields."""
    vals = np.fromstring(open(path).read(), sep=" ")
    return {
        "lat":   vals[0],   # latitude  (deg)
        "lon":   vals[1],   # longitude (deg)
        "alt":   vals[2],   # altitude  (m)
        "roll":  vals[3],   # roll      (rad)
        "pitch": vals[4],   # pitch     (rad)
        "yaw":   vals[5],   # heading   (rad)
    }


def _latlon_to_mercator(lat, lon, scale):
    """
    Convert lat/lon to Mercator XY in meters.

    Parameters
    ----------
    lat   : float   latitude in degrees
    lon   : float   longitude in degrees
    scale : float   Mercator scale factor = cos(lat_origin * pi/180)

    Returns
    -------
    x, y : float, float   Mercator coordinates in meters
    """
    x = scale * np.radians(lon) * EARTH_RADIUS
    y = scale * EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def _rpy_to_rotation(roll, pitch, yaw):
    """
    Build 3x3 rotation matrix from roll, pitch, yaw (rad).
    Rotation order: Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1,  0,   0 ],
                   [0,  cr, -sr],
                   [0,  sr,  cr]])

    Ry = np.array([[ cp, 0, sp],
                   [ 0,  1,  0],
                   [-sp, 0, cp]])

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,   0,  1]])

    return Rz @ Ry @ Rx


def _make_transform(R, t):
    """Build 4x4 homogeneous transform from R (3x3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# ------------------------------------------------------------------
# main converter
# ------------------------------------------------------------------

def oxts_to_poses(oxts_dir):
    """
    Convert KITTI raw OXTS data to 4x4 pose matrices in a local
    Cartesian frame anchored at frame 0.

    Parameters
    ----------
    oxts_dir : str
        Path to the oxts/data folder containing *.txt files.

    Returns
    -------
    poses : list of ndarray (4, 4)
        One pose per frame, expressed relative to frame 0.
        poses[0] is always identity.
    """
    oxts_files = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))

    if not oxts_files:
        raise FileNotFoundError("no .txt files found in %s" % oxts_dir)

    # read all oxts packets
    packets = [_read_oxts_file(f) for f in oxts_files]

    # mercator scale from origin latitude
    scale = np.cos(np.radians(packets[0]["lat"]))

    # origin in Mercator + altitude
    x0, y0 = _latlon_to_mercator(packets[0]["lat"], packets[0]["lon"], scale)
    z0      = packets[0]["alt"]

    # world frame pose at frame 0
    T0 = _make_transform(
        _rpy_to_rotation(packets[0]["roll"], packets[0]["pitch"], packets[0]["yaw"]),
        np.array([x0, y0, z0])
    )
    T0_inv = np.linalg.inv(T0)

    poses = []
    for pkt in packets:
        x, y = _latlon_to_mercator(pkt["lat"], pkt["lon"], scale)
        z    = pkt["alt"]
        R    = _rpy_to_rotation(pkt["roll"], pkt["pitch"], pkt["yaw"])
        T    = _make_transform(R, np.array([x, y, z]))

        # express relative to frame 0
        poses.append(T0_inv @ T)

    return poses


def poses_to_positions(poses):
    """Extract XYZ positions from a list of 4x4 pose matrices."""
    return np.array([T[:3, 3] for T in poses])


def save_poses(poses, path):
    """
    Save poses to a text file in KITTI format (12 numbers per line).

    Parameters
    ----------
    poses : list of ndarray (4, 4)
    path  : str   output file path
    """
    with open(path, "w") as f:
        for T in poses:
            row = T[:3, :].flatten()
            f.write(" ".join("%.6e" % v for v in row) + "\n")
    print("saved %d poses to %s" % (len(poses), path))


def load_poses_txt(path):
    """
    Load poses from a KITTI-format text file (12 numbers per line).
    Works for both our saved poses and the official poses/00.txt.

    Returns
    -------
    poses : list of ndarray (4, 4)
    """
    poses = []
    with open(path) as f:
        for line in f:
            vals = np.fromstring(line.strip(), sep=" ")
            T = np.eye(4)
            T[:3, :] = vals.reshape(3, 4)
            poses.append(T)
    return poses


# ------------------------------------------------------------------
# smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="OXTS to poses converter smoke test")
    parser.add_argument("oxts_dir",
                        help="path to oxts/data folder "
                             "(e.g. data/raw/2011_10_03/2011_10_03_drive_0027_sync/oxts/data/)")
    parser.add_argument("--save-poses", type=str, default=None,
                        help="save poses to this .txt file in KITTI format")
    parser.add_argument("--save-plot",  type=str, default=None,
                        help="save trajectory plot to this path (e.g. gt_traj.png)")
    parser.add_argument("--frames",     type=int, default=None,
                        help="only process first N frames (default: all)")
    args = parser.parse_args()

    print("loading OXTS data from: %s" % args.oxts_dir)
    poses = oxts_to_poses(args.oxts_dir)

    if args.frames:
        poses = poses[:args.frames]

    positions = poses_to_positions(poses)

    print("frames:          %d" % len(poses))
    print("total distance:  %.2f m" % np.sum(
        np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    print("XY range:        X=[%.1f, %.1f]  Y=[%.1f, %.1f]" % (
        positions[:, 0].min(), positions[:, 0].max(),
        positions[:, 1].min(), positions[:, 1].max()))
    print("pose[0]:\n%s" % np.round(poses[0], 6))
    print("pose[1]:\n%s" % np.round(poses[1], 6))

    if args.save_poses:
        save_poses(poses, args.save_poses)

    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(positions[:, 0], positions[:, 1],
            linewidth=1.2, color="steelblue", alpha=0.8)
    ax.scatter(positions[0,  0], positions[0,  1],
               c="green", s=80, zorder=5, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 1],
               c="red",   s=80, zorder=5, label="end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Ground Truth Trajectory (from OXTS)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        print("plot saved to: %s" % args.save_plot)

    plt.show()