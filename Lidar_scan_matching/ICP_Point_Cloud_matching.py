import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
def EstimateCorrespondence(X, Y, t, R, dmax):
    """
    Estimate point correspondences (Algorithm 1, lines 5–10).

    Parameters
    ----------
    X    : ndarray (nX, d) — source pointcloud
    Y    : ndarray (nY, d) — target pointcloud
    t    : ndarray (d, 1)  — current translation estimate
    R    : ndarray (d, d)  — current rotation estimate
    dmax : float           — max distance for correspondence

    Returns
    -------
    C : ndarray (K, 2) — correspondence pairs [(i, j), ...]
    """
    # TODO: For each point x_i in X:
    #   1. Compute transformed point: y_hat = R @ x_i + t
    #   2. Find closest point y_j in Y to y_hat
    #   3. If distance < dmax, add (i, j) to C
    C= []
    for i in range(len(X)):
        x_transformed = (R @ X[i].reshape(3,1) + t).ravel()
        dists = np.linalg.norm(Y - x_transformed, axis=1)
        j = np.argmin(dists)
        if dists[j] < dmax:
            C.append([i, j])
    return np.array(C)     
                
                 
    raise NotImplementedError("Implement EstimateCorrespondence")
def ComputeOptimalRigidRegistration(X, Y, C):
    """
    Compute optimal (R, t) aligning corresponding points (Algorithm 2).

    Parameters
    ----------
    X : ndarray (nX, d) — source pointcloud
    Y : ndarray (nY, d) — target pointcloud
    C : ndarray (K, 2)  — correspondence pairs

    Returns
    -------
    R : ndarray (d, d) — optimal rotation
    t : ndarray (d, 1) — optimal translation
    """
    # TODO: Implement Horn's method (Algorithm 2).
    # compute the centroid of the correspondence list 
   
    src_pts = X[C[:, 0]]   # (K, 3)
    tgt_pts = Y[C[:, 1]]   # (K, 3)

    # Centroids
    centroid_X = src_pts.mean(axis=0, keepdims=True).T  # (3, 1)
    centroid_Y = tgt_pts.mean(axis=0, keepdims=True).T  # (3, 1)

    # Demeaned 
    A = src_pts - centroid_X.T   # (K, 3)
    B = tgt_pts - centroid_Y.T   # (K, 3)

    W = A.T @ B   # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(W)

    # Fix 2: Reflection correction — ensures det(R) = +1
    D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T

    t = centroid_Y - R @ centroid_X

    return R, t
    raise NotImplementedError("Implement ComputeOptimalRigidRegistration")
def SE3_transform(X, R, t):
    """Apply rigid transformation (provided — do not modify)."""
    return (R @ X.T).T + t.T


def RMSE(X, Y, C):
    """Root-mean-squared error for corresponding points (provided — do not modify)."""
    return np.sqrt(np.linalg.norm(X[C[:, 0]] - Y[C[:, 1]], axis=1).mean())
def ICP(X, Y, t0, R0, dmax, num_ICP_iters):
    """
    Iterative Closest Point (Algorithm 1).

    Parameters
    ----------
    X              : ndarray (nX, d) — source pointcloud
    Y              : ndarray (nY, d) — target pointcloud
    t0             : ndarray (d, 1)  — initial translation
    R0             : ndarray (d, d)  — initial rotation
    dmax           : float           — max correspondence distance
    num_ICP_iters  : int             — number of iterations

    Returns
    -------
    R : ndarray (d, d) — estimated rotation
    t : ndarray (d, 1) — estimated translation
    C : ndarray (K, 2) — final correspondences
    """
    # TODO: Implement the ICP loop.
    #   t, R = t0, R0
    #   for each iteration:
    #       C = EstimateCorrespondence(X, Y, t, R, dmax)
    #       R, t = ComputeOptimalRigidRegistration(X, Y, C)
    #       (optionally print RMSE)
    #   return R, t, C
    t=t0.copy()
    R= R0.copy()
    for i in range(num_ICP_iters):
        C = EstimateCorrespondence(X,Y,t,R,dmax)
        R,t = ComputeOptimalRigidRegistration(X,Y,C)
    return R,t,C

    raise NotImplementedError("Implement ICP")
if __name__ == "__main__":
    
        pt_cld_X = pd.read_csv(osp.join(r"C:\Users\lenovo\Downloads\Mobile_Robotics\Assignment2\ICP scan matching\Data", "pclX.txt"),
                            header=None, names=["X", "Y", "Z"],
                            delimiter=" ").values
        pt_cld_Y = pd.read_csv(osp.join(r"C:\Users\lenovo\Downloads\Mobile_Robotics\Assignment2\ICP scan matching\Data", "pclY.txt"),
                            header=None, names=["X", "Y", "Z"],
                            delimiter=" ").values

        print(f"Pointcloud X: {pt_cld_X.shape}")
        print(f"Pointcloud Y: {pt_cld_Y.shape}")
        t = np.zeros ((3,1))
        R = np.eye(3)
        dmax = 0.25
        num_ICP_iters= 30
        # TODO: Run ICP with:
        #   R0 = I_3, t0 = 0, dmax = 0.25, num_ICP_iters = 30
        # Report:
        #   - Estimated R and t
        #   - Verify R^T R ≈ I and det(R) ≈ 1
        #   - RMSE
        #   - Plot co-registered pointclouds
        R_computed,t_computed,C_computed= ICP(pt_cld_X, pt_cld_Y, t, R, dmax, num_ICP_iters)
        if np.allclose(R_computed @ R_computed.T, np.eye(3)) and np.isclose(np.linalg.det(R_computed), 1.0):
                print("")
        X_transformed = SE3_transform(pt_cld_X, R_computed,t_computed)
        rmse = RMSE(X_transformed, pt_cld_Y, C_computed)
        print(f"The final RMSE value is :{rmse}")
        print(f"The rotation and translation after 30 iters is {R_computed}, {t_computed}")
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        
        # ax.scatter(pt_cld_X[:,0], pt_cld_X[:,1], pt_cld_X[:,2], 
        #         c='blue', label='Source (Original)', alpha=0.5)

        # Plot target
        ax.scatter(pt_cld_Y[:,0], pt_cld_Y[:,1], pt_cld_Y[:,2], 
                c='red', label='Target', alpha=0.5)

        # Plot transformed source
        ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2], 
                c='green', label='Source (Transformed)', alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('ICP Alignment of Point Clouds')
        ax.legend()

        plt.show()
        