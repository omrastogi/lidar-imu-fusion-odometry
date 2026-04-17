import numpy as np


#gravity in world frame
GRAVITY = np.array([0.0, 0.0, -9.81])


def skew(v):

    #skew symmetric matrix from 3-vector
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ]
    ])


def exp_so3(phi):

    #exponential map for SO(3), rodrigues formula
    angle = np.linalg.norm(phi)

    if angle < 1e-10:
        return np.eye(3) + skew(phi)

    axis = phi / angle
    K    = skew(axis)

    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K
    return R


def right_jacobian_so3(phi):

    #right jacobian of SO(3)
    angle = np.linalg.norm(phi)

    if angle < 1e-10:
        return np.eye(3) - 0.5 * skew(phi)

    axis = phi / angle
    K    = skew(axis)

    Jr = (np.eye(3)
          - (1.0 - np.cos(angle)) / (angle ** 2) * K
          + (angle - np.sin(angle)) / (angle ** 3) * K @ K)
    return Jr


class IMUPreintegrator:

    def __init__(self, bg=None, ba=None):

        #gyro and accel biases, default zero
        self.bg = bg if bg is not None else np.zeros(3)
        self.ba = ba if ba is not None else np.zeros(3)

        self.reset()

    def reset(self):

        #preintegrated measurements
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.dt_sum  = 0.0

        #15x15 covariance (rotation, velocity, position, bg, ba)
        self.cov = np.zeros((15, 15))

        #jacobians wrt biases for first order bias correction
        self.J_R_bg = np.zeros((3, 3))
        self.J_v_bg = np.zeros((3, 3))
        self.J_v_ba = np.zeros((3, 3))
        self.J_p_bg = np.zeros((3, 3))
        self.J_p_ba = np.zeros((3, 3))

        #noise parameters (continuous time)
        #typical values for automotive grade imu
        self.sigma_g  = 1.7e-4
        self.sigma_a  = 2.0e-3
        self.sigma_bg = 3.0e-6
        self.sigma_ba = 2.0e-5

    def integrate(self, gyro, accel, dt):

        #correct for bias
        gyro_corrected  = gyro - self.bg
        accel_corrected = accel - self.ba

        #rotation increment
        phi   = gyro_corrected * dt
        dR    = exp_so3(phi)
        Jr    = right_jacobian_so3(phi)

        #update preintegrated measurements
        #order matters, update p and v before R
        self.delta_p = (self.delta_p
                        + self.delta_v * dt
                        + 0.5 * self.delta_R @ accel_corrected * dt ** 2)

        self.delta_v = self.delta_v + self.delta_R @ accel_corrected * dt

        #propagate covariance
        self._propagate_covariance(accel_corrected, dt, dR, Jr)

        #propagate bias jacobians
        self._propagate_bias_jacobians(accel_corrected, dt, dR, Jr)

        #update rotation last
        self.delta_R = self.delta_R @ dR

        self.dt_sum += dt

    def _propagate_covariance(self, accel_corrected, dt, dR, Jr):

        #noise covariance (discrete time)
        Qi = np.zeros((12, 12))
        Qi[0:3, 0:3]   = (self.sigma_g ** 2) * dt * np.eye(3)
        Qi[3:6, 3:6]   = (self.sigma_a ** 2) * dt * np.eye(3)
        Qi[6:9, 6:9]   = (self.sigma_bg ** 2) * dt * np.eye(3)
        Qi[9:12, 9:12] = (self.sigma_ba ** 2) * dt * np.eye(3)

        #state transition matrix A (15x15)
        A = np.eye(15)

        A[0:3, 0:3]   = dR.T
        A[3:6, 0:3]   = -self.delta_R @ skew(accel_corrected) * dt
        A[3:6, 12:15] = -self.delta_R * dt
        A[6:9, 0:3]   = -0.5 * self.delta_R @ skew(accel_corrected) * dt ** 2
        A[6:9, 3:6]   = np.eye(3) * dt
        A[6:9, 12:15] = -0.5 * self.delta_R * dt ** 2

        #noise input matrix B (15x12)
        B = np.zeros((15, 12))
        B[0:3, 0:3]   = Jr * dt
        B[3:6, 3:6]   = self.delta_R * dt
        B[6:9, 3:6]   = 0.5 * self.delta_R * dt ** 2
        B[9:12, 6:9]  = np.eye(3) * dt
        B[12:15, 9:12] = np.eye(3) * dt

        self.cov = A @ self.cov @ A.T + B @ Qi @ B.T

    def _propagate_bias_jacobians(self, accel_corrected, dt, dR, Jr):

        #update jacobians for first order bias correction
        self.J_p_ba = self.J_p_ba + self.J_v_ba * dt - 0.5 * self.delta_R * dt ** 2
        self.J_p_bg = (self.J_p_bg + self.J_v_bg * dt
                       - 0.5 * self.delta_R @ skew(accel_corrected) @ self.J_R_bg * dt ** 2)

        self.J_v_ba = self.J_v_ba - self.delta_R * dt
        self.J_v_bg = self.J_v_bg - self.delta_R @ skew(accel_corrected) @ self.J_R_bg * dt

        self.J_R_bg = dR.T @ self.J_R_bg - Jr * dt

    def integrate_between(self, imu_measurements):

        #integrate a list of imu measurements (from get_imu_between)
        for m in imu_measurements:

            self.integrate(m["gyro"], m["accel"], m["dt"])

        return self.get_result()

    def get_result(self):

        return {
            "delta_R":  self.delta_R.copy(),
            "delta_v":  self.delta_v.copy(),
            "delta_p":  self.delta_p.copy(),
            "dt":       self.dt_sum,
            "cov":      self.cov.copy(),
            "J_R_bg":   self.J_R_bg.copy(),
            "J_v_bg":   self.J_v_bg.copy(),
            "J_v_ba":   self.J_v_ba.copy(),
            "J_p_bg":   self.J_p_bg.copy(),
            "J_p_ba":   self.J_p_ba.copy(),
        }


def propagate_state(R_prev, v_prev, p_prev, preint, dt):

    #propagate full state using preintegrated measurements
    #R_prev, v_prev, p_prev are the state at time k in world frame
    R_next = R_prev @ preint["delta_R"]
    v_next = v_prev + GRAVITY * dt + R_prev @ preint["delta_v"]
    p_next = p_prev + v_prev * dt + 0.5 * GRAVITY * dt ** 2 + R_prev @ preint["delta_p"]

    return R_next, v_next, p_next


def transform_to_velo(preint_result, T_imu_to_velo):
    """
    Re-express preintegrated IMU deltas in the LiDAR (Velodyne) frame.

    The preintegrated quantities (delta_R, delta_v, delta_p) are produced
    in the IMU body frame at the start of the integration window.  For
    LiDAR-odometry we need them expressed in the LiDAR frame instead.

    Parameters
    ----------
    preint_result : dict
        Output of IMUPreintegrator.get_result().
    T_imu_to_velo : ndarray (4,4)
        Rigid transform from IMU frame to LiDAR frame (KittiCalibration.T_imu_to_velo).

    Returns
    -------
    dict  — same keys as preint_result, with delta_R / delta_v / delta_p
            expressed in the LiDAR frame.  Covariance and Jacobians are
            rotated into the new frame as well.
    """
    R_IV = T_imu_to_velo[:3, :3]   # rotation: IMU -> LiDAR
    t_IV = T_imu_to_velo[:3,  3]   # translation: origin of IMU in LiDAR coords

    dR = preint_result["delta_R"]
    dv = preint_result["delta_v"]
    dp = preint_result["delta_p"]

    dR_velo = R_IV @ dR @ R_IV.T
    dv_velo = R_IV @ dv
    # lever-arm correction: displacement of LiDAR origin due to IMU rotation
    dp_velo = R_IV @ dp - (dR_velo - np.eye(3)) @ t_IV

    # rotate 3x3 jacobian blocks into new frame
    def _rot_jac(J):
        return R_IV @ J @ R_IV.T

    result = dict(preint_result)
    result["delta_R"] = dR_velo
    result["delta_v"] = dv_velo
    result["delta_p"] = dp_velo
    result["J_R_bg"]  = _rot_jac(preint_result["J_R_bg"])
    result["J_v_bg"]  = R_IV @ preint_result["J_v_bg"] @ R_IV.T
    result["J_v_ba"]  = R_IV @ preint_result["J_v_ba"]
    result["J_p_bg"]  = R_IV @ preint_result["J_p_bg"] @ R_IV.T
    result["J_p_ba"]  = R_IV @ preint_result["J_p_ba"]

    # rotate covariance: build block-diagonal rotation for (rot, vel, pos) sub-block
    P = preint_result["cov"].copy()
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            P[i:i+3, j:j+3] = R_IV @ preint_result["cov"][i:i+3, j:j+3] @ R_IV.T
    result["cov"] = P

    return result


if __name__ == "__main__":
    import argparse
    from kitti_loader import KittiRawLoader

    parser = argparse.ArgumentParser(description="IMU preintegration smoke test on KITTI data")
    parser.add_argument("drive_dir",
                        help="path to drive sync folder "
                             "(e.g. data/raw/2011_10_03/2011_10_03_drive_0027_sync/)")
    parser.add_argument("--frames", type=int, default=6,
                        help="number of consecutive LiDAR frames to test (default 6)")
    args = parser.parse_args()

    loader = KittiRawLoader(args.drive_dir)
    print("drive:        %s" % args.drive_dir)
    print("LiDAR frames: %d" % loader.n_frames)
    print("IMU samples:  %d" % loader.n_imu)
    if loader.calib:
        print("calibration:  loaded  (T_imu_to_velo available)")
        print("  T_imu_to_velo R:\n%s" % np.round(loader.calib.T_imu_to_velo[:3, :3], 6))
        print("  T_imu_to_velo t: %s m" % np.round(loader.calib.T_imu_to_velo[:3, 3], 6))
    else:
        print("calibration:  not found — results stay in IMU frame")
    print()

    n = min(args.frames, loader.n_frames) - 1
    preint = IMUPreintegrator()

    for k in range(n):
        preint.reset()
        imu_batch = loader.get_imu_between(k, k + 1)
        result    = preint.integrate_between(imu_batch)

        if loader.calib:
            result = transform_to_velo(result, loader.calib.T_imu_to_velo)
            frame_label = "LiDAR frame"
        else:
            frame_label = "IMU frame"

        t0 = loader.get_timestamp(k)
        t1 = loader.get_timestamp(k + 1)
        print("frames %d -> %d  (dt_lidar=%.4f s, dt_imu=%.4f s, n_imu=%d)  [%s]" % (
            k, k + 1, t1 - t0, result["dt"], len(imu_batch), frame_label))
        print("  delta_R (deg): %s" % str(np.round(
            np.degrees(np.array([result["delta_R"][2, 1],
                                 result["delta_R"][0, 2],
                                 result["delta_R"][1, 0]])), 4)))
        print("  delta_v (m/s): %s" % str(np.round(result["delta_v"], 4)))
        print("  delta_p   (m): %s" % str(np.round(result["delta_p"], 4)))