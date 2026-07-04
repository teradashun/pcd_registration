import numpy as np


def compute_pair_features(
    p1: np.ndarray, n1: np.ndarray, p2_array: np.ndarray, n2_array: np.ndarray
) -> np.ndarray:
    """1つの基準点と複数の近傍点とのペア特徴量を計算する

    Args:
        p1: 基準点の座標 (3,)
        n1: 基準点の法線 (3,)
        p2_array: 近傍点の座標配列 (num_neighbors, 3)
        n2_array: 近傍点の法線配列 (num_neighbors, 3)

    Returns:
        計算された特徴量（num_neighbors, 3） [theta, phi, alpha]
        # theta: 法線同士のねじれ具合（回転角）
        # phi: 2つの法線の傾き具合
        # alpha: 2点間の方向に対する法線の傾き具合
    """

    num_neighbors = p2_array.shape[0]
    result = np.zeros((num_neighbors, 3))

    # 1. 2点間のベクトル (diff_vectors) と距離を求める
    diff_vectors = p2_array - p1
    dist = np.linalg.norm(diff_vectors, axis=1)

    # 距離が0の要素（自分自身など）を弾くマスク
    valid_mask = dist > 0
    if not np.any(valid_mask):
        return result

    diff_vectors = diff_vectors[valid_mask]
    n2_valid = n2_array[valid_mask]
    dist_valid = dist[valid_mask]

    # 2. [基準の統一] 角度の小さい方を基準にする (swap処理)
    # angle1: p1からp2への方向と、n1がなす角のcos
    # angle2: p1からp2への方向と、n2がなす角のcos
    angle1 = np.dot(diff_vectors, n1) / dist_valid
    angle2 = np.sum(n2_valid * diff_vectors, axis=1) / dist_valid
    swap_mask = np.arccos(np.abs(angle1)) > np.arccos(np.abs(angle2))

    n1_copy = np.tile(n1, (len(dist_valid), 1))
    n2_copy = n2_valid.copy()

    n1_copy[swap_mask] = n2_valid[swap_mask]
    n2_copy[swap_mask] = n1
    diff_vectors[swap_mask] *= -1.0

    alpha = np.where(swap_mask, -angle2, angle1)

    # 3. [軸1の作成] 2点間のベクトル と n1 の外積からローカルX軸「v」を作る
    v = np.cross(diff_vectors, n1_copy)
    v_norm = np.linalg.norm(v, axis=1)

    # v_normが0のものはスキップ
    v_valid_mask = v_norm > 0
    if not np.any(v_valid_mask):
        return result

    # 有効なデータのみ抽出し、vを正規化（単位ベクトル化）する
    v = v[v_valid_mask] / v_norm[v_valid_mask][:, np.newaxis]
    n1_copy = n1_copy[v_valid_mask]
    n2_copy = n2_copy[v_valid_mask]
    alpha = alpha[v_valid_mask]

    # 4. [軸2の作成] n1 と v の外積からローカルY軸「w」を作る
    w = np.cross(n1_copy, v)

    # 5. [座標系の完成 ＆ 特徴量抽出]
    # Z軸=n1, X軸=v, Y軸=w のローカル座標系に対して、n2が向いている角度を計算
    phi = np.sum(v * n2_copy, axis=1)
    theta = np.arctan2(np.sum(w * n2_copy, axis=1), np.sum(n1_copy * n2_copy, axis=1))

    # 元の配列インデックスに書き戻す
    final_valid_indices = np.where(valid_mask)[0][v_valid_mask]
    result[final_valid_indices, 0] = theta
    result[final_valid_indices, 1] = phi
    result[final_valid_indices, 2] = alpha

    return result


def compute_spfh(
    source: np.ndarray, normals: np.ndarray, neighbors_list: list
) -> np.ndarray:
    """全点のSPFHを計算する

    Args:
        source: 点群座標 (num_source, 3)
        normals: 法線ベクトル (num_source, 3)
        neighbors_list: 各点の近傍点インデックスのリスト

    Returns:
        SPFH特徴量 (num_source, 33)
    """

    num_source = len(source)
    spfh = np.zeros((num_source, 33))

    for i in range(num_source):
        neighbors = neighbors_list[i]
        # 自分自身を除外
        valid_neighbors = [idx for idx in neighbors if idx != i]

        if len(valid_neighbors) == 0:
            continue

        p1 = source[i]
        n1 = normals[i]
        p2_array = source[valid_neighbors]
        n2_array = normals[valid_neighbors]

        pf = compute_pair_features(p1, n1, p2_array, n2_array)
        theta = pf[:, 0]
        phi = pf[:, 1]
        alpha = pf[:, 2]

        hist_incr = 100.0 / len(valid_neighbors)

        h_theta = np.floor(11 * (theta + np.pi) / (2.0 * np.pi)).astype(int)
        h_theta = np.clip(h_theta, 0, 10)

        h_phi = np.floor(11 * (phi + 1.0) * 0.5).astype(int)
        h_phi = np.clip(h_phi, 0, 10)

        h_alpha = np.floor(11 * (alpha + 1.0) * 0.5).astype(int)
        h_alpha = np.clip(h_alpha, 0, 10)

        spfh[i, 0:11] += np.bincount(h_theta, minlength=11) * hist_incr
        spfh[i, 11:22] += np.bincount(h_phi, minlength=11) * hist_incr
        spfh[i, 22:33] += np.bincount(h_alpha, minlength=11) * hist_incr

    return spfh


def compute_fpfh(
    source: np.ndarray, normals: np.ndarray, neighbors_list: list
) -> np.ndarray:
    """
    最終的なFPFH特徴量を計算する

    Args:
        source: 点群座標 (num_source, 3)
        normals: 法線ベクトル (num_source, 3)
        neighbors_list: 各点の近傍点インデックスのリスト

    Returns:
        FPFH特徴量 (num_source, 33)
    """

    num_source = len(source)
    fpfh = np.zeros((num_source, 33))

    spfh = compute_spfh(source, normals, neighbors_list)

    # 近傍点のSPFHを距離の重み付きで集約する
    for i in range(num_source):
        neighbors = neighbors_list[i]
        valid_neighbors = [idx for idx in neighbors if idx != i]

        if len(valid_neighbors) == 0:
            fpfh[i] = spfh[i]
            continue

        p1 = source[i]
        p2_array = source[valid_neighbors]

        dist_square = np.sum((p2_array - p1) ** 2, axis=1)

        neighbor_spfh = spfh[valid_neighbors]

        # 重み付き和の計算
        weighted_spfh = neighbor_spfh / dist_square[:, np.newaxis]
        sum_weighted_spfh = np.sum(weighted_spfh, axis=0)

        # 3つの特徴(11次元ずつ)の合計で割って100にする
        NUM_BINS = 11
        histogram_sums = np.zeros(3)
        histogram_sums[0] = np.sum(sum_weighted_spfh[0:NUM_BINS])
        histogram_sums[1] = np.sum(sum_weighted_spfh[NUM_BINS : NUM_BINS * 2])
        histogram_sums[2] = np.sum(sum_weighted_spfh[NUM_BINS * 2 : NUM_BINS * 3])

        # 0割りを防ぐための正規化係数
        norm_factor = np.zeros(3)
        for j in range(3):
            if histogram_sums[j] != 0.0:
                norm_factor[j] = 100.0 / histogram_sums[j]

        # 正規化した近傍SPFHに、自分自身のSPFHを足す
        final_feature = np.zeros(33)
        final_feature[0:11] = sum_weighted_spfh[0:11] * norm_factor[0] + spfh[i, 0:11]
        final_feature[11:22] = (
            sum_weighted_spfh[11:22] * norm_factor[1] + spfh[i, 11:22]
        )
        final_feature[22:33] = (
            sum_weighted_spfh[22:33] * norm_factor[2] + spfh[i, 22:33]
        )

        fpfh[i] = final_feature

    return fpfh
