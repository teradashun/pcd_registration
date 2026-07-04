import numpy as np
import open3d as o3d
from pathlib import Path
from collections import defaultdict


def download_data():
    """データをダウンロードする関数"""

    print("Stanford Bunny データを読み込み中")
    mesh = o3d.data.BunnyMesh()
    pcd = o3d.io.read_point_cloud(mesh.path)

    points = np.asarray(pcd.points)

    num_target_points = points.shape[0]
    target_colors = np.zeros((num_target_points, 3), dtype=np.float32)
    target_colors[:, 0] = 255  # 赤:（255, 0, 0）を付与
    pcd_data = np.hstack((points, target_colors))

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    target_path = data_dir / "bunny_target.txt"
    source_path = data_dir / "bunny_source.txt"

    np.savetxt(target_path, pcd_data, fmt="%.8f %.8f %.8f %d %d %d")
    print(f"Targetデータを保存しました: {target_path} (点数: {num_target_points})")

    # Sourceデータの作成
    # Z軸まわりに90度回転、X軸方向に0.05移動、y軸方向に0.1移動
    theta = np.radians(90)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    t = np.array([0.05, 0.1, 0.0])

    shifted_points = np.dot(points, R.T) + t

    num_source_points = shifted_points.shape[0]
    source_colors = np.zeros((num_source_points, 3), dtype=np.float32)
    source_colors[:, 2] = 255
    source_pcd_data = np.hstack((shifted_points, source_colors))

    np.savetxt(source_path, source_pcd_data, fmt="%.8f %.8f %.8f %d %d %d")
    print(f"Sourceデータを保存しました: {source_path} (点数: {num_source_points})")


def download_and_make_missing_data():
    """データをダウンロードする関数"""

    print("Stanford Bunny データを読み込み中")
    mesh = o3d.data.BunnyMesh()
    pcd = o3d.io.read_point_cloud(mesh.path)

    points = np.asarray(pcd.points)

    # 点群をスライスして欠損を作る
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    x_mid = (x_min + x_max) / 2.0

    # Target: 「左側〜中央やや右」までを残す
    target_mask = points[:, 0] < (x_mid + 0.02)
    target_points = points[target_mask]

    # Source: 「右側〜中央やや左」までを残す
    source_mask = points[:, 0] > (x_mid - 0.02)
    source_points_base = points[source_mask]

    num_target_points = target_points.shape[0]
    target_colors = np.zeros((num_target_points, 3), dtype=np.float32)
    target_colors[:, 0] = 255  # 赤:（255, 0, 0）を付与
    pcd_data = np.hstack((target_points, target_colors))

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    target_path = data_dir / "bunny_target.txt"
    source_path = data_dir / "bunny_source.txt"

    np.savetxt(target_path, pcd_data, fmt="%.8f %.8f %.8f %d %d %d")
    print(f"Targetデータを保存しました: {target_path} (点数: {num_target_points})")

    # Sourceデータの作成
    # Z軸まわりに90度回転、X軸方向に0.05移動、y軸方向に0.1移動
    theta = np.radians(90)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    t = np.array([0.05, 0.1, 0.0])

    shifted_points = np.dot(source_points_base, R.T) + t

    num_source_points = shifted_points.shape[0]
    source_colors = np.zeros((num_source_points, 3), dtype=np.float32)
    source_colors[:, 2] = 255
    source_pcd_data = np.hstack((shifted_points, source_colors))

    np.savetxt(source_path, source_pcd_data, fmt="%.8f %.8f %.8f %d %d %d")
    print(f"Sourceデータを保存しました: {source_path} (点数: {num_source_points})")


def search_hybrid(
    data: np.ndarray, query: np.ndarray, radius: float, max_neighbors: int
) -> list:
    """ハイブリッドサーチを行う（ボクセルグリッド空間ハッシュによる高速化）。

    Args:
        data: 探索対象の点群の座標を表すnumpy配列 (N, 3)
        query: 探索クエリの点群の座標を表すnumpy配列 (M, 3)
        radius: 探索半径
        max_neighbors: 各クエリ点に対して返す近傍点の最大数

    Returns:
        各クエリ点に対して、探索対象の点群の近傍点のインデックスを格納したリスト (M, max_neighbors)
    """

    if (
        data.size == 0
        or query.size == 0
        or data.shape[1] != 3
        or query.shape[1] != 3
        or max_neighbors <= 0
    ):
        return []

    # downsample_point_cloudのボクセルインデックスを再利用
    min_bound = np.min(data, axis=0)
    voxel_min_bound = min_bound - radius * 0.5

    data_voxels = np.floor((data - voxel_min_bound) / radius).astype(np.int32)
    query_voxels = np.floor((query - voxel_min_bound) / radius).astype(np.int32)

    voxel_dict = defaultdict(list)
    # NumPy配列のイテレーションは遅いため、リスト化してから回す
    for i, (vx, vy, vz) in enumerate(data_voxels.tolist()):
        voxel_dict[(vx, vy, vz)].append(i)

    voxel_dict = {k: np.array(v, dtype=np.int32) for k, v in voxel_dict.items()}

    # 探索対象となる周囲27近傍ボクセルの相対オフセット
    offsets = np.array(
        [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]],
        dtype=np.int32,
    )

    neighbors = []
    # query_voxels もリスト化した方が、forループ内のアンパック(qx, qy, qz)が高速
    query_voxels_list = query_voxels.tolist()

    query_num = len(query)
    explored_milestones = {
        int(query_num * 0.2): "20%",
        int(query_num * 0.4): "40%",
        int(query_num * 0.6): "60%",
        int(query_num * 0.8): "80%",
        query_num: "100%",
    }

    print("近傍探索開始")
    for i, q in enumerate(query):

        # 進捗の表示（20%, 40%, 60%, 80%, 100%）
        current_count = i + 1
        if current_count in explored_milestones:
            print(f"探索進捗: {explored_milestones[current_count]} 完了")

        qx, qy, qz = query_voxels_list[i]
        candidates = []

        for dx, dy, dz in offsets:
            key = (qx + dx, qy + dy, qz + dz)
            if key in voxel_dict:
                candidates.append(voxel_dict[key])

        if not candidates:
            neighbors.append([])
            continue

        candidate_indices = np.concatenate(candidates)
        candidate_points = data[candidate_indices]

        diff = candidate_points - q
        # np.einsum: 掛け算をしながら同時に足し算を行うため、メモリの消費が抑えられ、処理が高速になる
        dist_squared = np.einsum("ij,ij->i", diff, diff)

        valid_mask = dist_squared < radius**2
        valid_indices = candidate_indices[valid_mask]
        valid_dist_squared = dist_squared[valid_mask]

        if len(valid_indices) == 0:
            neighbors.append([])
            continue

        # 距離が近い順にソートして max_neighbors 件を取得
        if max_neighbors == 1:
            best_idx = np.argmin(valid_dist_squared)
            neighbors.append([valid_indices[best_idx]])

        elif len(valid_indices) <= max_neighbors:
            sorted_idx = np.argsort(valid_dist_squared)
            neighbors.append(valid_indices[sorted_idx].tolist())

        else:
            # np.argpartitionでmax_neighbors個に絞るため、処理が高速になる
            partition_idx = np.argpartition(valid_dist_squared, max_neighbors - 1)[
                :max_neighbors
            ]
            top_dists = valid_dist_squared[partition_idx]
            sort_top = np.argsort(top_dists)
            final_indices = partition_idx[sort_top]
            neighbors.append(
                valid_indices[final_indices].tolist()
            )  # クエリ点に距離が近い順に並ぶ

    return neighbors


def estimate_normals(points: np.ndarray, neighbors_list: list) -> np.ndarray:
    """法線推定を行う。

    Args:
        points: 法線を計算したい点群の座標 (M, 3)
        neighbors_list: 各点の近傍点のインデックスリスト (M個の要素を持つリスト)

    Returns:
        推定された法線ベクトル (M, 3)
    """
    normals = np.zeros_like(points)

    for i, neighbors in enumerate(neighbors_list):
        # 平面を定義するために、最低3点の近傍点を用意
        if len(neighbors) < 3:
            normals[i] = np.array([0.0, 0.0, 1.0])
            continue

        neighbor_points = points[neighbors]

        # 共分散行列の計算
        mean = np.mean(neighbor_points, axis=0)
        centered = neighbor_points - mean
        covariance = np.dot(centered.T, centered) / len(neighbors)

        # 固有値分解
        _, eigenvectors = np.linalg.eigh(covariance)

        # 最小の固有値に対応する固有ベクトルを取得
        normal = eigenvectors[:, 0]

        # 単位ベクトル化
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm

        if normal[2] < 0:
            normal = -normal

        normals[i] = normal

    return normals
