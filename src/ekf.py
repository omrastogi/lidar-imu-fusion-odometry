import numpy as np
from imu_integrator import (
    IMUPreintegrator,
    transform_to_velo,
    propagate_state,
    exp_so3,
    skew,
    GRAVITY,
)


# ------------------------------------------------------------------
# SO3 log map -- inverse of exp_so3
# ------------------------------------------------------------------

def log_so3(R):
    """
    SO(3) logarithm map: rotation matrix -> axis-angle vector (3,).
    Inverse of exp_so3.
    """
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle     = np.arccos(cos_angle)

    if angle < 1e-10:
        return np.zeros(3)

    log = angle / (2.0 * np.sin(angle)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return log


# ------------------------------------------------------------------
# EKF
# ------------------------------------------------------------------

class LidarImuEKF:
    """
    15-state EKF for LiDAR-IMU fusion.

    State vector (error state formulation):
        delta_x = [delta_phi(3), delta_v(3), delta_p(3), delta_bg(3), delta_ba(3)]

    Nominal state (stored separately):
        R  : ndarray (3,3)   orientation in world frame
        v  : ndarray (3,)    velocity in world frame
        p  : ndarray (3,)    position in world frame
        bg : ndarray (3,)    gyroscope bias
        ba : ndarray (3,)    accelerometer bias

    Usage
    -----
        ekf = LidarImuEKF()

        # per frame:
        T_pred = ekf.predict(imu_batch, calib)   # -> 4x4 for ICP init
        T_fused = ekf.update(T_icp)              # -> 4x4 fused pose
        R, v, p = ekf.get_state()
    """

    def __init__(self, sigma_r=0.01, sigma_t=0.05, v_init=None):
        """
        Parameters
        ----------
        sigma_r : float         ICP rotation noise std (rad)
        sigma_t : float         ICP translation noise std (m)
        v_init  : ndarray (3,)  initial velocity in world/LiDAR frame (m/s).
                                Provide from OXTS for accurate IMU prediction.
        """
        # nominal state
        self.R  = np.eye(3)
        self.v  = v_init.copy() if v_init is not None else np.zeros(3)
        self.p  = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)

        # error state covariance (15x15)
        # P_vel small because velocity is initialized from OXTS (~0.1 m/s accuracy)
        # P_pos large because absolute position is unknown; ICP provides relative updates
        p_vel = 0.01 if v_init is not None else 1.0
        self.P = np.diag([
            1e-4, 1e-4, 1e-4,        # delta_phi  (rad^2)
            p_vel, p_vel, p_vel,     # delta_v    ((m/s)^2)
            100.,  100.,  100.,      # delta_p    (m^2)
            1e-6,  1e-6,  1e-6,     # delta_bg   ((rad/s)^2)
            1e-4,  1e-4,  1e-4,     # delta_ba   ((m/s^2)^2)
        ])

        # measurement noise (6x6) -- rotation (3) + translation (3)
        self.R_meas = np.zeros((6, 6))
        self.R_meas[0:3, 0:3] = np.eye(3) * sigma_r ** 2
        self.R_meas[3:6, 3:6] = np.eye(3) * sigma_t ** 2

        # store last preintegration result for update step
        self._last_preint   = None
        self._last_dt       = None
        self._R_k           = np.eye(3)   # R before predict -- needed for H matrix
        self._last_t_pred_b = None        # full predicted body-frame displacement

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, imu_batch, calib=None, dt_lidar=None):
        """
        IMU predict step.

        Parameters
        ----------
        imu_batch : list of dict   from loader.get_imu_between(k, k+1)
        calib     : KittiCalibration or None
        dt_lidar  : float or None  actual LiDAR inter-frame interval (seconds).
                    When provided, state propagation and the F[pos,vel] coupling
                    use this interval instead of the IMU integration window, which
                    is critical when fewer IMU samples are available than expected.

        Returns
        -------
        T_predicted : ndarray (4, 4)
            Predicted relative transform in LiDAR frame -- use as ICP T_init.
        """
        if not imu_batch:
            return np.eye(4)

        self._R_k = self.R.copy()   # save R_k before nominal state update

        preint = IMUPreintegrator(bg=self.bg.copy(), ba=self.ba.copy())
        result = preint.integrate_between(imu_batch)

        if calib is not None:
            result = transform_to_velo(result, calib.T_imu_to_velo)

        self._last_preint = result
        self._last_dt     = result["dt"]

        # use LiDAR inter-frame interval for state propagation when given;
        # this is necessary because OXTS in the sync folder may have only
        # one sample per LiDAR frame, leaving part of the interval uncovered
        dt_prop = dt_lidar if dt_lidar is not None else result["dt"]

        # propagate nominal state
        p_before = self.p.copy()
        R_next, v_next, p_next = propagate_state(
            self.R, self.v, self.p, result, dt_prop
        )
        # full body-frame predicted displacement = what the nominal state predicts for ICP
        self._last_t_pred_b = self._R_k.T @ (p_next - p_before)

        # propagate error state covariance
        F = np.eye(15)

        # rotation block: delta_phi propagation
        F[0:3, 0:3]   = result["delta_R"].T
        F[0:3, 9:12]  = -result["J_R_bg"]

        # velocity block
        F[3:6, 0:3]   = -self.R @ skew(result["delta_v"])
        F[3:6, 9:12]  = -self.R @ result["J_v_bg"]
        F[3:6, 12:15] = -self.R @ result["J_v_ba"]

        # position block -- use dt_prop for v->p coupling to match ICP coverage
        F[6:9, 0:3]   = -self.R @ skew(result["delta_p"])
        F[6:9, 3:6]   = np.eye(3) * dt_prop
        F[6:9, 9:12]  = -self.R @ result["J_p_bg"]
        F[6:9, 12:15] = -self.R @ result["J_p_ba"]

        # propagate covariance
        Q       = result["cov"]
        self.P  = F @ self.P @ F.T + Q

        # build predicted relative transform for ICP init
        # ICP uses source->target convention: R_est = delta_R.T, t_est = -delta_p
        T_predicted         = np.eye(4)
        T_predicted[:3, :3] = result["delta_R"].T
        T_predicted[:3,  3] = -result["delta_p"]

        # update nominal state to predicted
        self.R = R_next
        self.v = v_next
        self.p = p_next

        return T_predicted

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, T_icp):
        """
        LiDAR/ICP measurement update step.

        Parameters
        ----------
        T_icp : ndarray (4, 4)
            Relative transform from ICP (source -> target), in LiDAR frame.

        Returns
        -------
        T_fused : ndarray (4, 4)
            Fused pose in world frame after update.
        """
        # ICP gives source->target: R_icp = delta_R_body.T, t_icp = -delta_p_body
        # Convert to body-frame displacement convention to match IMU preintegration
        R_meas = T_icp[:3, :3].T
        t_meas = -T_icp[:3,  3]

        # predicted relative transform -- use full nominal state prediction (v*dt + IMU)
        if self._last_preint is not None:
            R_pred = self._last_preint["delta_R"]
        else:
            R_pred = np.eye(3)

        if self._last_t_pred_b is not None:
            t_pred = self._last_t_pred_b   # R_k.T @ (p_{k+1|k} - p_k): full body-frame prediction
        elif self._last_preint is not None:
            t_pred = self._last_preint["delta_p"]
        else:
            t_pred = np.zeros(3)

        # innovation -- difference between ICP and IMU prediction
        # rotation innovation: log(R_pred.T @ R_meas)
        dR  = R_pred.T @ R_meas
        phi = log_so3(dR)

        # translation innovation
        dt  = t_meas - t_pred

        # stack into 6D innovation vector
        y = np.concatenate([phi, dt])

        # measurement Jacobian H (6x15)
        # maps error state to measurement space
        H           = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)        # rotation innovation from rotation error
        # ICP translation is in LiDAR/body frame; position error state is world frame
        H[3:6, 6:9] = self._R_k.T      # rotate world-frame position error to body frame

        # innovation covariance
        S = H @ self.P @ H.T + self.R_meas

        # Kalman gain (15x6)
        K = self.P @ H.T @ np.linalg.inv(S)

        print("  innov y: %s  K_norm: %.4f" % (np.round(y, 4), np.linalg.norm(K)))

        # error state correction
        delta_x = K @ y

        # inject correction into nominal state
        delta_phi = delta_x[0:3]
        delta_v   = delta_x[3:6]
        delta_p   = delta_x[6:9]
        delta_bg  = delta_x[9:12]
        delta_ba  = delta_x[12:15]

        self.R  = self.R  @ exp_so3(delta_phi)
        self.v  = self.v  + delta_v
        self.p  = self.p  + delta_p
        self.bg = self.bg + delta_bg
        self.ba = self.ba + delta_ba

        # update covariance -- Joseph form for numerical stability
        I_KH    = np.eye(15) - K @ H
        self.P  = I_KH @ self.P @ I_KH.T + K @ self.R_meas @ K.T

        # return fused world frame pose
        return self._get_world_pose()

    # ------------------------------------------------------------------
    # state accessors
    # ------------------------------------------------------------------

    def get_state(self):
        """Return current nominal state (R, v, p)."""
        return self.R.copy(), self.v.copy(), self.p.copy()

    def get_pose(self):
        """Return current pose as 4x4 transform in world frame."""
        return self._get_world_pose()

    def _get_world_pose(self):
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3,  3] = self.p
        return T


# ------------------------------------------------------------------
# full pipeline
# ------------------------------------------------------------------

def run_ekf_pipeline(loader, n_frames=None,
                     voxel_size=0.3, max_dist=1.0, max_iters=30,
                     sigma_r=0.01, sigma_t=0.05):
    """
    Run full LiDAR-IMU EKF pipeline over a sequence.

    Returns
    -------
    positions : ndarray (n_frames, 3)
    T_chain   : list of ndarray (4,4)
    rmse_list : list of float
    """
    from scan_matching_icp import match_scans

    if n_frames is None:
        n_frames = loader.n_frames
    n_frames = min(n_frames, loader.n_frames)

    # bootstrap: read initial velocity from OXTS so IMU prediction starts accurate
    calib = loader.calib
    v_init = None
    if loader.oxts_files:
        vals0  = np.fromstring(open(loader.oxts_files[0]).read(), sep=" ")
        v_body = vals0[8:11]   # vf, vl, vu in IMU body frame
        if calib is not None:
            v_init = calib.T_imu_to_velo[:3, :3] @ v_body
        else:
            v_init = v_body.copy()

    ekf          = LidarImuEKF(sigma_r=sigma_r, sigma_t=sigma_t, v_init=v_init)
    positions    = [np.zeros(3)]
    T_chain      = [np.eye(4)]
    rmse_list    = []
    prev_tgt_pcd = None

    for k in range(n_frames - 1):

        # predict -- get IMU-based init for ICP
        dt_lidar    = loader.get_timestamp(k + 1) - loader.get_timestamp(k)
        imu_batch   = loader.get_imu_between(k, k + 1)
        T_predicted = ekf.predict(imu_batch, calib, dt_lidar=dt_lidar)

        # ICP with IMU predicted init, reusing previous target as source
        T_icp, rmse, n_corr, prev_tgt_pcd = match_scans(
            loader, k, k + 1,
            T_init=T_predicted,
            voxel_size=voxel_size,
            max_dist=max_dist,
            max_iters=max_iters,
            src_pcd=prev_tgt_pcd,   # reuse previous target
            tgt_pcd=None,           # always load fresh target
        )

        # update -- fuse ICP with IMU prediction
        T_fused = ekf.update(T_icp)

        positions.append(T_fused[:3, 3].copy())
        T_chain.append(T_fused.copy())
        rmse_list.append(rmse)

        print("frame %4d -> %4d  rmse=%.4f  corr=%5d" % (
            k, k + 1, rmse, n_corr))

    return np.array(positions), T_chain, rmse_list


# ------------------------------------------------------------------
# smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="LiDAR-IMU EKF smoke test")
    parser.add_argument("drive_dir",
                        help="path to KITTI raw drive sync folder")
    parser.add_argument("--frames",     type=int,   default=50,
                        help="number of frames to process (default 50)")
    parser.add_argument("--voxel-size", type=float, default=0.3)
    parser.add_argument("--max-dist",   type=float, default=1.0)
    parser.add_argument("--max-iters",  type=int,   default=30)
    parser.add_argument("--sigma-r",    type=float, default=0.01,
                        help="ICP rotation noise std in rad (default 0.01)")
    parser.add_argument("--sigma-t",    type=float, default=0.05,
                        help="ICP translation noise std in m (default 0.05)")
    parser.add_argument("--gt-oxts",    type=str,   default=None,
                        help="path to oxts/data folder for ground truth overlay")
    parser.add_argument("--save-plot",  type=str,   default=None)
    args = parser.parse_args()

    from kitti_loader import KittiRawLoader

    loader = KittiRawLoader(args.drive_dir)

    print("drive:       %s" % args.drive_dir)
    print("frames:      %d" % args.frames)
    print("voxel size:  %.2f m" % args.voxel_size)
    print("sigma_r:     %.4f rad" % args.sigma_r)
    print("sigma_t:     %.4f m"   % args.sigma_t)
    print()

    positions, T_chain, rmse_list = run_ekf_pipeline(
        loader,
        n_frames=args.frames,
        voxel_size=args.voxel_size,
        max_dist=args.max_dist,
        max_iters=args.max_iters,
        sigma_r=args.sigma_r,
        sigma_t=args.sigma_t,
    )

    print()
    print("EKF pipeline summary:")
    print("  frames processed : %d" % len(positions))
    print("  total distance   : %.2f m" % np.sum(
        np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    print("  mean RMSE        : %.4f m" % np.mean(rmse_list))
    print("  max  RMSE        : %.4f m" % np.max(rmse_list))

    # ground truth
    gt_positions = None
    if args.gt_oxts:
        from oxts_to_poses import oxts_to_poses, poses_to_positions
        gt_poses     = oxts_to_poses(args.gt_oxts)[:args.frames]
        gt_positions = poses_to_positions(gt_poses)
        print("  ground truth loaded: %d poses" % len(gt_poses))

    # plot
    n_cols = 2
    fig, axes = plt.subplots(1, n_cols, figsize=(14, 6))

    ax = axes[0]
    if gt_positions is not None:
        ax.plot(gt_positions[:, 0], gt_positions[:, 1],
                linewidth=1.2, color="gray", alpha=0.6,
                linestyle="--", label="ground truth")
    ax.plot(positions[:, 0], positions[:, 1],
            linewidth=1.2, color="steelblue", alpha=0.8, label="EKF fused")
    ax.scatter(positions[0,  0], positions[0,  1], c="green", s=80, zorder=5, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 1], c="red",   s=80, zorder=5, label="end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("LiDAR-IMU EKF Trajectory (%d frames)" % len(positions))
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(rmse_list, color="coral", linewidth=1.2)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("ICP RMSE (m)")
    ax2.set_title("ICP RMSE per frame")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        print("plot saved to: %s" % args.save_plot)

    plt.show()