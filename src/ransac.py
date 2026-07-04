import numpy as np
from icp import estimate_rigid_transform


def find_feature_matches(
    source_fpfh: np.ndarray, target_fpfh: np.ndarray
) -> np.ndarray:
    """FPFH特徴量のL2距離に基づいて、Sourceの各点に最も特徴が似ているTargetの点を見つける。

    Args:
        source_fpfh: Source点群のFPFH特徴量 (N, 33)
        target_fpfh: Target点群のFPFH特徴量 (M, 33)

    Returns:
        (N, 2)の配列。各行は [sourceのインデックス, targetのインデックス]
    """
    matches = []
    for i in range(len(source_fpfh)):
        diff = target_fpfh - source_fpfh[i]
        dist_sq = np.sum(diff**2, axis=1)
        best_idx = np.argmin(dist_sq)  # 特徴量距離が最小のTarget点
        matches.append([i, best_idx])
    return np.array(matches)


def ransac_feature_match(
    source: np.ndarray,
    target: np.ndarray,
    matches: np.ndarray,
    max_iterations: int,
    inlier_threshold: float,
) -> np.ndarray:
    """RANSACを用いて、特徴量マッチングの対応点から大まかな初期変換行列を推定する。

    Args:
        source: Source点群の座標データ (N, 3)
        target: Target点群の座標データ (M, 3)
        matches: 特徴量マッチングの結果得られた対応インデックスの配列 (K, 2)
        max_iterations: RANSACの最大反復回数
        inlier_threshold: インライアと判定するための距離の閾値

    Returns: 推定された4x4の同次変換行列 (4, 4)

    """
    best_inlier_count = 0
    best_T = np.identity(4)

    num_matches = len(matches)
    if num_matches < 3:
        return best_T

    # 対応した座標だけを抽出
    source_matched = source[matches[:, 0]]
    target_matched = target[matches[:, 1]]

    for _ in range(max_iterations):
        # ランダムに3つの対応点ペアをサンプリング
        sample_indices = np.random.choice(num_matches, 3, replace=False)
        src_samples = source_matched[sample_indices]
        tgt_samples = target_matched[sample_indices]

        # 3点から仮の変換行列を推定
        T_temp = estimate_rigid_transform(src_samples, tgt_samples)

        # 推定した行列で Source の対応点を全て変換
        transformed = np.dot(source_matched, T_temp[:3, :3].T) + T_temp[:3, 3]

        # インライア数をカウント
        dist_sq = np.sum((transformed - target_matched) ** 2, axis=1)
        inlier_count = np.sum(dist_sq < inlier_threshold**2)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_T = T_temp

    print(f"RANSAC インライア数: {best_inlier_count} / {num_matches}")
    return best_T
