import os
import numpy as np
import matplotlib.pyplot as plt
from kitti_loader import KittiLoader


def main():

    import argparse

    parser = argparse.ArgumentParser(description="KITTI loader sanity check")
    parser.add_argument("sequences_dir", help="path to sequences/ folder")
    parser.add_argument("--sequence", "-s", default="00", help="sequence id (default: 00)")
    args = parser.parse_args()

    sequences_dir = args.sequences_dir
    sequence      = args.sequence

    print("loading sequence %s..." % sequence)
    loader = KittiLoader(sequences_dir, sequence=sequence)

    n = loader.n_frames
    print("frames:       %d" % n)
    print("has calib:    %s" % (loader.calib is not None))
    print("has timestamps: %s" % (loader.timestamps is not None))
    print("has gt poses: %s" % loader.has_ground_truth())

    if loader.timestamps is not None:
        total_time = loader.get_timestamp(n - 1) - loader.get_timestamp(0)
        print("duration:     %.1f s" % total_time)

    #verify first scan loads
    scan = loader.get_lidar_scan(0)
    print("scan 0:       %s points" % scan.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("KITTI sequence %s — sanity check" % sequence)

    # panel 1: ground truth trajectory (if available)
    if loader.has_ground_truth():

        gt_x = [loader.get_ground_truth_pose(k)[0, 3] for k in range(n)]
        gt_y = [loader.get_ground_truth_pose(k)[1, 3] for k in range(n)]

        axes[0].plot(gt_x, gt_y, "b-", linewidth=1.5)
        axes[0].plot(gt_x[0],  gt_y[0],  "go", markersize=8, label="start")
        axes[0].plot(gt_x[-1], gt_y[-1], "rs", markersize=8, label="end")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[0].set_title("ground truth trajectory")
        axes[0].legend()
        axes[0].set_aspect("equal")
        axes[0].grid(True, alpha=0.3)

    else:
        axes[0].text(0.5, 0.5, "no ground truth\nfor sequence %s" % sequence,
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("ground truth trajectory")

    # panel 2: top-down view of scan 0
    pts = loader.get_lidar_scan(0)
    x, y = pts[:, 0], pts[:, 1]
    dist = np.sqrt(x ** 2 + y ** 2)
    mask = dist < 50.0  # clip to 50 m for visibility

    axes[1].scatter(x[mask], y[mask], s=0.3, c=dist[mask], cmap="viridis")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_title("scan 0 — top-down (50 m radius)")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = "sanity_check_seq%s.png" % sequence
    plt.savefig(out, dpi=150)
    plt.close()
    print("saved %s" % out)


if __name__ == "__main__":
    main()
