"""
evaluation_plots.py
====================
Plots LiDAR-only, IMU-only, and fused EKF trajectories.
Produces:
  - Figure 1: Combined comparison (all 3 overlaid)
  - Figure 2: LiDAR-only analysis
  - Figure 3: IMU-only analysis
  - Figure 4: EKF fused analysis

Usage:
    python evaluation_plots.py <drive_dir> --frames 500
    python evaluation_plots.py <drive_dir> --frames 500 --gt-oxts <path/to/oxts/data>
    python evaluation_plots.py <drive_dir> --frames 500 --save-plot "C:/path/to/results/plot.png"
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- import existing functions as-is --------------------------------
from kitti_loader      import KittiRawLoader
from scan_matching_icp import build_trajectory
from imu_integrator    import build_imu_trajectory
from ekf               import run_ekf_pipeline
from oxts_to_poses     import oxts_to_poses, poses_to_positions


# -----------------------------------------------------------------------
# plot-only helpers
# -----------------------------------------------------------------------

def _cumdist(positions):
    return np.concatenate([[0.0],
           np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])


def _trans_err(pos, gt):
    n = min(len(pos), len(gt))
    return np.linalg.norm(pos[:n] - gt[:n], axis=1)


def _apply_theme():
    plt.rcParams.update({
        'font.family':       'monospace',
        'axes.facecolor':    '#0d1117',
        'figure.facecolor':  '#0d1117',
        'axes.edgecolor':    '#30363d',
        'axes.labelcolor':   '#e6edf3',
        'xtick.color':       '#8b949e',
        'ytick.color':       '#8b949e',
        'text.color':        '#e6edf3',
        'grid.color':        '#21262d',
        'grid.linewidth':    0.8,
        'axes.titlesize':    10,
        'axes.labelsize':    9,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,
        'legend.framealpha': 0.25,
        'legend.edgecolor':  '#30363d',
    })


def _save(fig, base_path, suffix):
    if base_path:
        root, ext = os.path.splitext(base_path)
        path = f"{root}_{suffix}{ext}"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")


# -----------------------------------------------------------------------
# Figure 1 — combined comparison
# -----------------------------------------------------------------------

def plot_combined(lidar_pos, imu_pos, ekf_pos, frames, times,
                  lidar_dist, imu_dist, ekf_dist,
                  lidar_rmse, ekf_rmse,
                  gt_pos, has_gt,
                  lidar_err, imu_err, ekf_err,
                  save_path):

    C_LIDAR = '#ffa657'
    C_IMU   = '#d2a8ff'
    C_EKF   = '#58a6ff'
    C_GT    = '#3fb950'
    C_WARN  = '#f78166'

    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                            top=0.93, bottom=0.07, left=0.06, right=0.97)
    fig.suptitle(f"Combined Comparison: LiDAR / IMU / EKF  |  {len(frames)} frames  |  {times[-1]:.0f} s",
                 fontsize=13, fontweight='bold', color='#e6edf3', y=0.97)

    # [0, 0:2] trajectory overlay
    ax = fig.add_subplot(gs[0, :2])
    if has_gt:
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color=C_GT, lw=1.5, alpha=0.5,
                linestyle='--', label='Ground Truth')
    ax.plot(lidar_pos[:, 0], lidar_pos[:, 1], color=C_LIDAR, lw=1.2, alpha=0.85, label='LiDAR only')
    ax.plot(imu_pos[:,   0], imu_pos[:,   1], color=C_IMU,   lw=1.2, alpha=0.85, label='IMU only')
    ax.plot(ekf_pos[:,   0], ekf_pos[:,   1], color=C_EKF,   lw=1.8, alpha=0.95, label='EKF fused')
    ax.scatter(*ekf_pos[0,  :2], c=C_GT,   s=80, zorder=6, marker='o', edgecolors='white', lw=0.8, label='Start')
    ax.scatter(*ekf_pos[-1, :2], c=C_WARN, s=80, zorder=6, marker='X', edgecolors='white', lw=0.8, label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison')
    ax.set_aspect('equal'); ax.grid(True); ax.legend()

    # [0, 2] ICP RMSE
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(frames[1:len(lidar_rmse)+1], lidar_rmse, color=C_LIDAR, lw=1.0, alpha=0.8, label='LiDAR ICP')
    ax.plot(frames[1:len(ekf_rmse)+1],   ekf_rmse,   color=C_EKF,   lw=1.0, alpha=0.9, label='EKF ICP')
    ax.axhline(np.mean(lidar_rmse), color=C_LIDAR, lw=0.7, linestyle=':', alpha=0.5)
    ax.axhline(np.mean(ekf_rmse),   color=C_EKF,   lw=0.7, linestyle=':', alpha=0.5)
    ax.set_xlabel('Frame'); ax.set_ylabel('ICP RMSE (m)')
    ax.set_title('ICP RMSE Comparison')
    ax.grid(True); ax.legend()

    # [1, 0] error vs time
    ax = fig.add_subplot(gs[1, 0])
    if has_gt:
        ax.plot(times, lidar_err, color=C_LIDAR, lw=1.3, label='LiDAR only')
        ax.plot(times, imu_err,   color=C_IMU,   lw=1.3, label='IMU only')
        ax.plot(times, ekf_err,   color=C_EKF,   lw=1.8, label='EKF fused')
        ax.axhline(2.0, color=C_WARN, lw=0.9, linestyle='--', alpha=0.7, label='2 m limit')
        ax.set_ylabel('Position Error (m)')
    else:
        ax.plot(frames[1:], np.linalg.norm(np.diff(lidar_pos, axis=0), axis=1) * 10, color=C_LIDAR, lw=1.0, label='LiDAR')
        ax.plot(frames[1:], np.linalg.norm(np.diff(imu_pos,   axis=0), axis=1) * 10, color=C_IMU,   lw=1.0, label='IMU')
        ax.plot(frames[1:], np.linalg.norm(np.diff(ekf_pos,   axis=0), axis=1) * 10, color=C_EKF,   lw=1.4, label='EKF')
        ax.set_ylabel('Speed (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Goal 1: Error vs Time' if has_gt else 'Estimated Speed')
    ax.grid(True); ax.legend()

    # [1, 1] error vs distance
    ax = fig.add_subplot(gs[1, 1])
    if has_gt:
        ax.plot(lidar_dist, lidar_err, color=C_LIDAR, lw=1.3, label='LiDAR only')
        ax.plot(imu_dist,   imu_err,   color=C_IMU,   lw=1.3, label='IMU only')
        ax.plot(ekf_dist,   ekf_err,   color=C_EKF,   lw=1.8, label='EKF fused')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Goal 2: Drift Accumulation')
    else:
        ax.plot(frames, lidar_dist, color=C_LIDAR, lw=1.2, label='LiDAR')
        ax.plot(frames, imu_dist,   color=C_IMU,   lw=1.2, label='IMU')
        ax.plot(frames, ekf_dist,   color=C_EKF,   lw=1.6, label='EKF')
        ax.set_ylabel('Cumulative Distance (m)')
        ax.set_title('Cumulative Distance')
    ax.set_xlabel('Distance (m)' if has_gt else 'Frame')
    ax.grid(True); ax.legend()

    # [1, 2] drift rate or RMSE
    ax = fig.add_subplot(gs[1, 2])
    if has_gt:
        ax.plot(lidar_dist, lidar_err / np.maximum(lidar_dist, 1e-3) * 100, color=C_LIDAR, lw=1.2, label='LiDAR')
        ax.plot(imu_dist,   imu_err   / np.maximum(imu_dist,   1e-3) * 100, color=C_IMU,   lw=1.2, label='IMU')
        ax.plot(ekf_dist,   ekf_err   / np.maximum(ekf_dist,   1e-3) * 100, color=C_EKF,   lw=1.6, label='EKF')
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('Drift Rate (%)')
        ax.set_title('Drift Rate (error / distance × 100)')
    else:
        ax.plot(frames[1:len(lidar_rmse)+1], lidar_rmse, color=C_LIDAR, lw=1.0, label='LiDAR')
        ax.plot(frames[1:len(ekf_rmse)+1],   ekf_rmse,   color=C_EKF,   lw=1.0, label='EKF')
        ax.set_xlabel('Frame'); ax.set_ylabel('RMSE (m)')
        ax.set_title('ICP RMSE')
    ax.grid(True); ax.legend()

    # [2, 0] pose jumps
    ax = fig.add_subplot(gs[2, 0])
    lidar_step = np.linalg.norm(np.diff(lidar_pos, axis=0), axis=1)
    imu_step   = np.linalg.norm(np.diff(imu_pos,   axis=0), axis=1)
    ekf_step   = np.linalg.norm(np.diff(ekf_pos,   axis=0), axis=1)
    ax.plot(frames[1:], lidar_step, color=C_LIDAR, lw=0.9, alpha=0.7, label='LiDAR')
    ax.plot(frames[1:], imu_step,   color=C_IMU,   lw=0.9, alpha=0.7, label='IMU')
    ax.plot(frames[1:], ekf_step,   color=C_EKF,   lw=1.2, alpha=0.9, label='EKF')
    thr = np.mean(ekf_step) + 3 * np.std(ekf_step)
    ax.axhline(thr, color=C_WARN, lw=0.9, linestyle='--', label='jump thresh (μ+3σ)')
    ax.set_xlabel('Frame'); ax.set_ylabel('Step (m)')
    ax.set_title('Goal 3: Pose Jump Detection')
    ax.grid(True); ax.legend()

    # [2, 1] X divergence
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(frames, lidar_pos[:, 0], color=C_LIDAR, lw=1.0, linestyle='--', alpha=0.8, label='LiDAR X')
    ax.plot(frames, imu_pos[:,   0], color=C_IMU,   lw=1.0, linestyle=':',  alpha=0.8, label='IMU X')
    ax.plot(frames, ekf_pos[:,   0], color=C_EKF,   lw=1.4,                  alpha=0.9, label='EKF X')
    if has_gt:
        ax.plot(frames, gt_pos[:, 0], color=C_GT, lw=0.9, linestyle='-.', alpha=0.6, label='GT X')
    ax.set_xlabel('Frame'); ax.set_ylabel('X position (m)')
    ax.set_title('X-axis Divergence')
    ax.grid(True); ax.legend()

    # [2, 2] summary bar
    ax  = fig.add_subplot(gs[2, 2])
    labels = ['LiDAR\nonly', 'IMU\nonly', 'EKF\nfused']
    colors = [C_LIDAR, C_IMU, C_EKF]
    values = ([lidar_err[-1], imu_err[-1], ekf_err[-1]] if has_gt
              else [lidar_dist[-1], imu_dist[-1], ekf_dist[-1]])
    ylabel = 'Final Position Error (m)' if has_gt else 'Total Distance (m)'
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='#30363d', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='#e6edf3')
    ax.set_ylabel(ylabel); ax.set_title('Summary'); ax.grid(True, axis='y')

    _save(fig, save_path, 'combined')


# -----------------------------------------------------------------------
# Figure 2 — LiDAR only
# -----------------------------------------------------------------------

def plot_lidar(lidar_pos, frames, times, lidar_dist, lidar_rmse,
               gt_pos, has_gt, lidar_err, save_path):

    C = '#ffa657'; C_GT = '#3fb950'; C_WARN = '#f78166'

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            top=0.93, bottom=0.07, left=0.06, right=0.97)
    fig.suptitle(f"LiDAR-only (ICP)  |  {len(frames)} frames  |  {times[-1]:.0f} s",
                 fontsize=13, fontweight='bold', color='#e6edf3', y=0.97)

    ax = fig.add_subplot(gs[0, :2])
    if has_gt:
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color=C_GT, lw=1.5, alpha=0.5,
                linestyle='--', label='Ground Truth')
    ax.plot(lidar_pos[:, 0], lidar_pos[:, 1], color=C, lw=1.5, label='LiDAR only')
    ax.scatter(*lidar_pos[0,  :2], c=C_GT,   s=80, zorder=6, marker='o', edgecolors='white', lw=0.8, label='Start')
    ax.scatter(*lidar_pos[-1, :2], c=C_WARN, s=80, zorder=6, marker='X', edgecolors='white', lw=0.8, label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('LiDAR Trajectory'); ax.set_aspect('equal'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.fill_between(frames[1:len(lidar_rmse)+1], lidar_rmse, alpha=0.25, color=C)
    ax.plot(frames[1:len(lidar_rmse)+1], lidar_rmse, color=C, lw=1.2)
    ax.axhline(np.mean(lidar_rmse), color='white', lw=0.8, linestyle=':', alpha=0.6,
               label=f'mean={np.mean(lidar_rmse):.3f}')
    ax.set_xlabel('Frame'); ax.set_ylabel('ICP RMSE (m)')
    ax.set_title('ICP RMSE per Frame'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    if has_gt:
        ax.fill_between(times, lidar_err, alpha=0.2, color=C)
        ax.plot(times, lidar_err, color=C, lw=1.4)
        ax.axhline(2.0, color=C_WARN, lw=0.9, linestyle='--', label='2 m limit')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Error vs Time')
    else:
        speed = np.linalg.norm(np.diff(lidar_pos, axis=0), axis=1) * 10
        ax.fill_between(frames[1:], speed, alpha=0.2, color=C)
        ax.plot(frames[1:], speed, color=C, lw=1.2)
        ax.set_ylabel('Speed (m/s)'); ax.set_title('Estimated Speed')
    ax.set_xlabel('Time (s)' if has_gt else 'Frame'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    if has_gt:
        ax.fill_between(lidar_dist, lidar_err, alpha=0.2, color=C)
        ax.plot(lidar_dist, lidar_err, color=C, lw=1.4)
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('Position Error (m)')
        ax.set_title('Drift vs Distance')
    else:
        ax.fill_between(frames, lidar_dist, alpha=0.2, color=C)
        ax.plot(frames, lidar_dist, color=C, lw=1.4)
        ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative Distance (m)')
        ax.set_title('Cumulative Distance')
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 2])
    step = np.linalg.norm(np.diff(lidar_pos, axis=0), axis=1)
    thr  = np.mean(step) + 3 * np.std(step)
    bad  = np.where(step > thr)[0]
    ax.plot(frames[1:], step, color=C, lw=1.0)
    ax.axhline(thr, color=C_WARN, lw=0.9, linestyle='--', label=f'{len(bad)} jumps')
    if len(bad):
        ax.scatter(bad + 1, step[bad], c=C_WARN, s=25, zorder=5)
    ax.set_xlabel('Frame'); ax.set_ylabel('Step (m)')
    ax.set_title('Pose Jumps / Failures'); ax.grid(True); ax.legend()

    _save(fig, save_path, 'lidar')


# -----------------------------------------------------------------------
# Figure 3 — IMU only
# -----------------------------------------------------------------------

def plot_imu(imu_pos, frames, times, imu_dist,
             gt_pos, has_gt, imu_err, save_path):

    C = '#d2a8ff'; C_GT = '#3fb950'; C_WARN = '#f78166'

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            top=0.93, bottom=0.07, left=0.06, right=0.97)
    fig.suptitle(f"IMU-only (Dead Reckoning)  |  {len(frames)} frames  |  {times[-1]:.0f} s",
                 fontsize=13, fontweight='bold', color='#e6edf3', y=0.97)

    ax = fig.add_subplot(gs[0, :2])
    if has_gt:
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color=C_GT, lw=1.5, alpha=0.5,
                linestyle='--', label='Ground Truth')
    ax.plot(imu_pos[:, 0], imu_pos[:, 1], color=C, lw=1.5, label='IMU only')
    ax.scatter(*imu_pos[0,  :2], c=C_GT,   s=80, zorder=6, marker='o', edgecolors='white', lw=0.8, label='Start')
    ax.scatter(*imu_pos[-1, :2], c=C_WARN, s=80, zorder=6, marker='X', edgecolors='white', lw=0.8, label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('IMU Trajectory (dead reckoning)')
    ax.set_aspect('equal'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    speed = np.linalg.norm(np.diff(imu_pos, axis=0), axis=1) * 10
    ax.fill_between(frames[1:], speed, alpha=0.25, color=C)
    ax.plot(frames[1:], speed, color=C, lw=1.2)
    ax.set_xlabel('Frame'); ax.set_ylabel('Speed (m/s)')
    ax.set_title('IMU Estimated Speed'); ax.grid(True)

    ax = fig.add_subplot(gs[1, 0])
    if has_gt:
        ax.fill_between(times, imu_err, alpha=0.2, color=C)
        ax.plot(times, imu_err, color=C, lw=1.4)
        ax.axhline(2.0, color=C_WARN, lw=0.9, linestyle='--', label='2 m limit')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Error vs Time\n(IMU drift grows fast)')
    else:
        ax.fill_between(times, imu_dist, alpha=0.2, color=C)
        ax.plot(times, imu_dist, color=C, lw=1.4)
        ax.set_ylabel('Distance (m)')
        ax.set_title('Cumulative Distance vs Time')
    ax.set_xlabel('Time (s)'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    if has_gt:
        ax.fill_between(imu_dist, imu_err, alpha=0.2, color=C)
        ax.plot(imu_dist, imu_err, color=C, lw=1.4)
        drift_pct = imu_err / np.maximum(imu_dist, 1e-3) * 100
        ax2 = ax.twinx()
        ax2.plot(imu_dist, drift_pct, color='white', lw=0.8, linestyle=':', alpha=0.5)
        ax2.set_ylabel('Drift Rate (%)', color='#8b949e')
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('Position Error (m)')
        ax.set_title('IMU Drift vs Distance')
    else:
        ax.fill_between(frames, imu_dist, alpha=0.2, color=C)
        ax.plot(frames, imu_dist, color=C, lw=1.4)
        ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative Distance (m)')
        ax.set_title('Cumulative Distance')
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(frames, imu_pos[:, 0], color=C,       lw=1.0, label='X')
    ax.plot(frames, imu_pos[:, 1], color='#3fb950', lw=1.0, label='Y')
    ax.plot(frames, imu_pos[:, 2], color='#ffa657', lw=1.0, label='Z')
    if has_gt:
        ax.plot(frames, gt_pos[:, 0], color=C,       lw=0.7, linestyle='--', alpha=0.4)
        ax.plot(frames, gt_pos[:, 1], color='#3fb950', lw=0.7, linestyle='--', alpha=0.4)
    ax.set_xlabel('Frame'); ax.set_ylabel('Position (m)')
    ax.set_title('IMU X/Y/Z  (dashed = GT)'); ax.grid(True); ax.legend()

    _save(fig, save_path, 'imu')


# -----------------------------------------------------------------------
# Figure 4 — EKF fused
# -----------------------------------------------------------------------

def plot_ekf(ekf_pos, frames, times, ekf_dist, ekf_rmse,
             gt_pos, has_gt, ekf_err, save_path):

    C = '#58a6ff'; C_GT = '#3fb950'; C_WARN = '#f78166'

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            top=0.93, bottom=0.07, left=0.06, right=0.97)
    fig.suptitle(f"EKF Fused (LiDAR + IMU)  |  {len(frames)} frames  |  {times[-1]:.0f} s",
                 fontsize=13, fontweight='bold', color='#e6edf3', y=0.97)

    ax = fig.add_subplot(gs[0, :2])
    if has_gt:
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color=C_GT, lw=1.5, alpha=0.5,
                linestyle='--', label='Ground Truth')
    ax.plot(ekf_pos[:, 0], ekf_pos[:, 1], color=C, lw=1.5, label='EKF fused')
    ax.scatter(*ekf_pos[0,  :2], c=C_GT,   s=80, zorder=6, marker='o', edgecolors='white', lw=0.8, label='Start')
    ax.scatter(*ekf_pos[-1, :2], c=C_WARN, s=80, zorder=6, marker='X', edgecolors='white', lw=0.8, label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('EKF Fused Trajectory'); ax.set_aspect('equal'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ekf_rmse_arr = np.array(ekf_rmse)
    ax.fill_between(frames[1:len(ekf_rmse_arr)+1], ekf_rmse_arr, alpha=0.25, color=C)
    ax.plot(frames[1:len(ekf_rmse_arr)+1], ekf_rmse_arr, color=C, lw=1.2)
    thr = np.mean(ekf_rmse_arr) + 2 * np.std(ekf_rmse_arr)
    ax.axhline(np.mean(ekf_rmse_arr), color='white', lw=0.8, linestyle=':', alpha=0.6,
               label=f'mean={np.mean(ekf_rmse_arr):.3f}')
    ax.axhline(thr, color=C_WARN, lw=0.8, linestyle='--', alpha=0.7, label='μ+2σ')
    ax.set_xlabel('Frame'); ax.set_ylabel('ICP RMSE (m)')
    ax.set_title('EKF ICP RMSE per Frame'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    if has_gt:
        ax.fill_between(times, ekf_err, alpha=0.2, color=C)
        ax.plot(times, ekf_err, color=C, lw=1.4)
        ax.axhline(2.0, color=C_WARN, lw=0.9, linestyle='--', label='2 m limit')
        safe = np.where(ekf_err > 2.0)[0]
        if len(safe):
            ax.axvline(safe[0] / 10.0, color=C_WARN, lw=1.0, linestyle='-.',
                       label=f'safe window: {safe[0]/10:.0f}s')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Goal 1: Error vs Time')
    else:
        speed = np.linalg.norm(np.diff(ekf_pos, axis=0), axis=1) * 10
        ax.fill_between(frames[1:], speed, alpha=0.2, color=C)
        ax.plot(frames[1:], speed, color=C, lw=1.2)
        ax.set_ylabel('Speed (m/s)'); ax.set_title('EKF Estimated Speed')
    ax.set_xlabel('Time (s)' if has_gt else 'Frame'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    if has_gt:
        ax.fill_between(ekf_dist, ekf_err, alpha=0.2, color=C)
        ax.plot(ekf_dist, ekf_err, color=C, lw=1.4)
        if ekf_dist[-1] > 5:
            coef = np.polyfit(ekf_dist, ekf_err, 1)
            ax.plot(ekf_dist, np.polyval(coef, ekf_dist), color='white',
                    lw=0.9, linestyle=':', alpha=0.7,
                    label=f'drift≈{coef[0]*100:.2f}%')
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('Position Error (m)')
        ax.set_title('Goal 2: Drift Accumulation')
    else:
        ax.fill_between(frames, ekf_dist, alpha=0.2, color=C)
        ax.plot(frames, ekf_dist, color=C, lw=1.4)
        ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative Distance (m)')
        ax.set_title('Cumulative Distance')
    ax.grid(True); ax.legend()

    ax = fig.add_subplot(gs[1, 2])
    step = np.linalg.norm(np.diff(ekf_pos, axis=0), axis=1)
    thr  = np.mean(step) + 3 * np.std(step)
    bad  = np.where(step > thr)[0]
    ax.plot(frames[1:], step, color=C, lw=1.0)
    ax.axhline(thr, color=C_WARN, lw=0.9, linestyle='--', label=f'{len(bad)} jumps (μ+3σ)')
    if len(bad):
        ax.scatter(bad + 1, step[bad], c=C_WARN, s=25, zorder=5)
    ax.set_xlabel('Frame'); ax.set_ylabel('Step (m)')
    ax.set_title('Goal 3: EKF Pose Jumps'); ax.grid(True); ax.legend()

    _save(fig, save_path, 'ekf')


# -----------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('drive_dir')
    parser.add_argument('--frames',     type=int,   default=500)
    parser.add_argument('--voxel-size', type=float, default=0.3)
    parser.add_argument('--max-dist',   type=float, default=1.0)
    parser.add_argument('--max-iters',  type=int,   default=30)
    parser.add_argument('--sigma-r',    type=float, default=0.01)
    parser.add_argument('--sigma-t',    type=float, default=0.05)
    parser.add_argument('--gt-oxts',    type=str,   default=None)
    parser.add_argument('--save-plot',  type=str,   default=None)
    args = parser.parse_args()

    loader = KittiRawLoader(args.drive_dir)
    N      = min(args.frames, loader.n_frames)

    # ---- run pipelines ----------------------------------------------
    print("\n[1/3] LiDAR-only ICP")
    lidar_pos, _, lidar_rmse = build_trajectory(
        loader, n_frames=N,
        voxel_size=args.voxel_size,
        max_dist=args.max_dist,
        max_iters=args.max_iters,
    )

    print("\n[2/3] IMU-only dead reckoning")
    imu_pos = build_imu_trajectory(loader, n_frames=N)

    print("\n[3/3] Fused EKF")
    ekf_pos, _, ekf_rmse = run_ekf_pipeline(
        loader, n_frames=N,
        voxel_size=args.voxel_size,
        max_dist=args.max_dist,
        max_iters=args.max_iters,
        sigma_r=args.sigma_r,
        sigma_t=args.sigma_t,
    )

    # ---- align lengths ----------------------------------------------
    N         = min(len(lidar_pos), len(imu_pos), len(ekf_pos))
    lidar_pos = lidar_pos[:N];  imu_pos = imu_pos[:N];  ekf_pos = ekf_pos[:N]
    frames    = np.arange(N)
    times     = frames / 10.0

    lidar_dist = _cumdist(lidar_pos)
    imu_dist   = _cumdist(imu_pos)
    ekf_dist   = _cumdist(ekf_pos)
    lidar_rmse = np.array(lidar_rmse)
    ekf_rmse   = np.array(ekf_rmse)

    # ---- ground truth -----------------------------------------------
    gt_pos = None
    oxts_src = args.gt_oxts or loader.oxts_dir
    try:
        gt_pos = poses_to_positions(oxts_to_poses(oxts_src)[:N])
        print(f"\nGround truth loaded: {len(gt_pos)} poses")
    except Exception as e:
        print(f"\nNo ground truth: {e}")

    has_gt    = gt_pos is not None and len(gt_pos) >= N
    lidar_err = imu_err = ekf_err = None
    if has_gt:
        gt_pos    = gt_pos[:N]
        lidar_err = _trans_err(lidar_pos, gt_pos)
        imu_err   = _trans_err(imu_pos,   gt_pos)
        ekf_err   = _trans_err(ekf_pos,   gt_pos)
        print(f"\nFinal errors — LiDAR: {lidar_err[-1]:.2f}m  "
              f"IMU: {imu_err[-1]:.2f}m  EKF: {ekf_err[-1]:.2f}m")

    # ---- apply theme and plot ---------------------------------------
    _apply_theme()

    print("\nGenerating plots...")

    plot_combined(lidar_pos, imu_pos, ekf_pos, frames, times,
                  lidar_dist, imu_dist, ekf_dist,
                  lidar_rmse, ekf_rmse,
                  gt_pos, has_gt,
                  lidar_err, imu_err, ekf_err,
                  args.save_plot)

    plot_lidar(lidar_pos, frames, times, lidar_dist, lidar_rmse,
               gt_pos, has_gt, lidar_err, args.save_plot)

    plot_imu(imu_pos, frames, times, imu_dist,
             gt_pos, has_gt, imu_err, args.save_plot)

    plot_ekf(ekf_pos, frames, times, ekf_dist, ekf_rmse,
             gt_pos, has_gt, ekf_err, args.save_plot)

    plt.show()


if __name__ == '__main__':
    main()