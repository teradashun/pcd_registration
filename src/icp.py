import numpy as np


def register_centroids(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """2つの点群の重心を計算し、初期位置を合わせるための4x4変換行列を返す。

    Args:
        source: 移動させたい点群 (N, 3)
        target: 基準となる点群 (M, 3)

    Returns:
        重心を合わせるための4x4変換行列
    """

    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)

    translation = target_centroid - source_centroid

    T = np.identity(4)
    T[:3, 3] = translation
    return T


def compute_rmse(source: np.ndarray, target: np.ndarray, neighbors_list: list) -> float:
    """ダウンサンプリングしていない元の点群全体に対して、最も近い点との距離のRMSEを計算する。

    Args:
        source: 変換後のsource点群 (N, 3)
        target: target点群 (M, 3)
        neighbors_list: sourceの各点に対するtargetの最寄りのインデックスリスト

    Returns:
        RMSE値（対応点が見つからない場合は float("inf")）
    """

    distances_sq = []
    for i, p in enumerate(source):
        if len(neighbors_list[i]) == 0:
            continue
        closest_target_pt = target[neighbors_list[i][0]]
        distances_sq.append(np.sum((p - closest_target_pt) ** 2))

    # 探索半径内に対応点が見つからない場合、誤差を無限大とする
    if len(distances_sq) == 0:
        return float("inf")

    return float(np.sqrt(np.mean(distances_sq)))


def estimate_rigid_transform(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """対応点ペアから、最小二乗法を用いて最良の回転・平行移動行列を計算する。

    Args:
        source: 移動前の点群 (K, 3)
        destination: 移動後の対応点群 (K, 3)

    Returns:
        4x4の変換行列
    """

    if len(source) == 0 or len(destination) == 0:
        return np.identity(4)

    source_centroid = np.mean(source, axis=0)
    destination_centroid = np.mean(destination, axis=0)

    source_centered = source - source_centroid
    destination_centered = destination - destination_centroid

    # 共分散行列の計算
    H = np.dot(source_centered.T, destination_centered)

    # 特異値分解
    U, _, Vt = np.linalg.svd(H)

    # 回転行列の計算
    R = np.dot(Vt.T, U.T)

    # 反転のチェックと修正
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 平行移動ベクトルの計算
    t = destination_centroid - np.dot(R, source_centroid)

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def estimate_point_to_plane_transform(
    source: np.ndarray, destination: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Point-to-Plane誤差を最小化する、最良の回転・平行移動行列を計算する。

    Args:
        source: 移動前の点群 (K, 3)
        destination: 移動後の対応点群 (K, 3)
        normals: destination点群の法線ベクトル (K, 3)

    Returns:
        4x4の変換行列
    """

    if len(source) < 3 or len(destination) < 3:
        return np.identity(4)

    # 微小回転を仮定した線形方程式 Ax = b を構築
    # x = [alpha, beta, gamma, tx, ty, tz]^T

    c = np.cross(source, normals)
    A = np.hstack((c, normals))
    b = np.sum((destination - source) * normals, axis=1)

    # 最小二乗法で方程式を解く: (A^T * A)x = A^T * b
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # 解から回転角と平行移動を抽出
    alpha, beta, gamma, tx, ty, tz = x

    # オイラー角から回転行列Rを復元
    cx, cy, cz = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sx, sy, sz = np.sin(alpha), np.sin(beta), np.sin(gamma)

    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    R = R_z @ R_y @ R_x

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    return T
