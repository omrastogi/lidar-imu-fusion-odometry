import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse


# ------------------------------------------------------------------
# point cloud helpers
# ------------------------------------------------------------------

def numpy_to_o3d(points):
    """Convert Nx3 numpy array to open3d PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def preprocess(points, voxel_size=0.3):
    """Voxel downsample and estimate normals (needed for point-to-plane ICP)."""
    pcd = numpy_to_o3d(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd


# ------------------------------------------------------------------
# core ICP -- works on preprocessed open3d point clouds
# ------------------------------------------------------------------

def run_icp_pcd(src_pcd, tgt_pcd, T_init=None, max_dist=1.0, max_iters=30):
    """
    Run Open3D point-to-plane ICP on preprocessed point clouds.

    Parameters
    ----------
    src_pcd  : o3d.geometry.PointCloud   source (already downsampled + normals)
    tgt_pcd  : o3d.geometry.PointCloud   target (already downsampled + normals)
    T_init   : ndarray (4, 4)            initial transform (default: identity)
    max_dist : float                     max correspondence distance
    max_iters: int                       max ICP iterations

    Returns
    -------
    T_est  : ndarray (4, 4)   estimated transform (source -> target)
    rmse   : float            inlier RMSE after alignment
    n_corr : int              number of correspondences used
    """
    if T_init is None:
        T_init = np.eye(4) 

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=max_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters),
    )

    return (
        result.transformation,
        result.inlier_rmse,
        len(result.correspondence_set),
    )


# ------------------------------------------------------------------
# scan matching
# ------------------------------------------------------------------

def match_scans(loader, k_start, k_end, T_init=None,
                voxel_size=0.3, max_dist=1.0, max_iters=30,
                src_pcd=None, tgt_pcd=None):
    """
    Match two consecutive LiDAR scans.

    If src_pcd / tgt_pcd are provided, reuse them (avoids redundant preprocessing).
    Otherwise load and preprocess internally.

    Parameters
    ----------
    loader   : KittiRawLoader
    k_start  : int                       source frame index
    k_end    : int                       target frame index
    T_init   : ndarray (4, 4)            initial transform (default: identity)
                                         pass EKF predicted pose here when available
    src_pcd  : o3d.geometry.PointCloud   preprocessed source (optional)
    tgt_pcd  : o3d.geometry.PointCloud   preprocessed target (optional)

    Returns
    -------
    T_est   : ndarray (4, 4)
    rmse    : float
    n_corr  : int
    tgt_pcd : o3d.geometry.PointCloud    preprocessed target (reuse as next src)
    """
    if src_pcd is None:
        src_pcd = preprocess(loader.get_lidar_scan(k_start)[:, :3], voxel_size)

    if tgt_pcd is None:
        tgt_pcd = preprocess(loader.get_lidar_scan(k_end)[:, :3], voxel_size)

    T_est, rmse, n_corr = run_icp_pcd(
        src_pcd, tgt_pcd,
        T_init=T_init,
        max_dist=max_dist,
        max_iters=max_iters,
    )

    return T_est, rmse, n_corr, tgt_pcd


# ------------------------------------------------------------------
# trajectory builder (pure ICP, no fusion)
# ------------------------------------------------------------------

def build_trajectory(loader, n_frames=None,
                     voxel_size=0.3, max_dist=1.0, max_iters=30):
    """
    Chain ICP results across frames to build a full trajectory.
    Reuses preprocessed point clouds -- each scan is preprocessed exactly once.

    Returns
    -------
    positions : ndarray (n_frames, 3)   XYZ position at each frame
    T_chain   : list of ndarray (4,4)   cumulative transforms
    rmse_list : list of float
    """
    if n_frames is None:
        n_frames = loader.n_frames

    n_frames     = min(n_frames, loader.n_frames)
    T_global     = np.eye(4)
    positions    = [T_global[:3, 3].copy()]
    T_chain      = [T_global.copy()]
    rmse_list    = []
    prev_tgt_pcd = None

    for k in range(n_frames - 1):

        # reuse previous target as current source -- no redundant preprocessing
        if k == 0:
            src_pcd = preprocess(loader.get_lidar_scan(k)[:, :3], voxel_size)
        else:
            src_pcd = prev_tgt_pcd

        tgt_pcd = preprocess(loader.get_lidar_scan(k + 1)[:, :3], voxel_size)

        T_est, rmse, n_corr, prev_tgt_pcd = match_scans(
            loader, k, k + 1,
            voxel_size=voxel_size,
            max_dist=max_dist,
            max_iters=max_iters,
            src_pcd=src_pcd,
            tgt_pcd=tgt_pcd,
        )

        # T_global = T_global @ T_est
        T_global = T_global @ np.linalg.inv(T_est)

        positions.append(T_global[:3, 3].copy())
        T_chain.append(T_global.copy())
        rmse_list.append(rmse)

        print("frame %4d -> %4d  rmse=%.4f  corr=%5d" % (
            k, k + 1, rmse, n_corr))

    return np.array(positions), T_chain, rmse_list


# ------------------------------------------------------------------
# trajectory plot
# ------------------------------------------------------------------

def plot_trajectory(positions, rmse_list=None, gt_positions=None,
                    title="LiDAR Odometry Trajectory", save_path=None):
    """
    Plot the estimated trajectory (top-down XY view).

    Parameters
    ----------
    positions    : ndarray (N, 3)   estimated positions
    rmse_list    : list of float    ICP RMSE per frame (optional)
    gt_positions : ndarray (N, 3)   ground truth positions (optional)
    save_path    : str              save figure to disk (optional)
    """
    n_cols = 2 if rmse_list else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(14 if n_cols == 2 else 8, 6))

    if n_cols == 1:
        axes = [axes]

    ax = axes[0]

    # ground truth
    if gt_positions is not None:
        ax.plot(gt_positions[:, 0], gt_positions[:, 1],
                linewidth=1.2, color="gray", alpha=0.6,
                label="ground truth", linestyle="--")

    # estimated
    ax.plot(positions[:, 0], positions[:, 1],
            linewidth=1.2, color="steelblue", alpha=0.8, label="estimated")
    ax.scatter(positions[0,  0], positions[0,  1], c="green", s=80, zorder=5, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 1], c="red",   s=80, zorder=5, label="end")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE over frames
    if rmse_list:
        ax2 = axes[1]
        ax2.plot(rmse_list, color="coral", linewidth=1.2)
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("ICP RMSE (m)")
        ax2.set_title("ICP RMSE per frame")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("trajectory plot saved to: %s" % save_path)

    plt.show()


# ------------------------------------------------------------------
# smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scan matching smoke test")
    parser.add_argument("drive_dir",
                        help="path to KITTI raw drive sync folder "
                             "(e.g. data/raw/2011_10_03/2011_10_03_drive_0027_sync/)")
    parser.add_argument("--frames",     type=int,   default=50,
                        help="number of frames to process (default 50)")
    parser.add_argument("--voxel-size", type=float, default=0.3,
                        help="voxel downsample size in meters (default 0.3)")
    parser.add_argument("--max-dist",   type=float, default=1.0,
                        help="max ICP correspondence distance (default 1.0)")
    parser.add_argument("--max-iters",  type=int,   default=30,
                        help="max ICP iterations (default 30)")
    parser.add_argument("--save-plot",  type=str,   default=None,
                        help="save trajectory plot to this path (e.g. traj.png)")
    parser.add_argument("--gt-oxts",    type=str,   default=None,
                        help="path to oxts/data folder for ground truth overlay")
    args = parser.parse_args()

    from kitti_loader import KittiRawLoader

    loader = KittiRawLoader(args.drive_dir)

    print("drive:       %s" % args.drive_dir)
    print("frames:      %d" % args.frames)
    print("voxel size:  %.2f m" % args.voxel_size)
    print()

    positions, T_chain, rmse_list = build_trajectory(
        loader,
        n_frames=args.frames,
        voxel_size=args.voxel_size,
        max_dist=args.max_dist,
        max_iters=args.max_iters,
    )

    print()
    print("trajectory summary:")
    print("  frames processed : %d" % len(positions))
    print("  total distance   : %.2f m" % np.sum(
        np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    print("  mean RMSE        : %.4f m" % np.mean(rmse_list))
    print("  max  RMSE        : %.4f m" % np.max(rmse_list))

    # load ground truth -- explicit path wins, else fall back to loader's oxts dir
    gt_positions = None
    oxts_src = args.gt_oxts or loader.oxts_dir
    try:
        from oxts_to_poses import oxts_to_poses, poses_to_positions
        gt_poses     = oxts_to_poses(oxts_src)[:args.frames]
        gt_positions = poses_to_positions(gt_poses)
        print("  ground truth loaded: %d poses" % len(gt_poses))
    except Exception as e:
        print("  ground truth unavailable: %s" % e)

    plot_trajectory(
        positions,
        rmse_list=rmse_list,
        gt_positions=gt_positions,
        title="LiDAR Odometry (%d frames)" % len(positions),
        save_path=args.save_plot,
    )
    print("ICP positions:\n", positions)
    print("GT positions:\n", gt_positions[:5])
