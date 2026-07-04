import numpy as np


def downsample_point_cloud(
    point_coords: np.ndarray, point_colors: np.ndarray, voxel_size: float
) -> tuple:
    """点群をダウンサンプリングする。

    Args:
        point_coords: 点群の座標を表すnumpy配列 (N, 3)
        point_colors: 点群の色を表すnumpy配列 (N, 3)
        voxel_size: ダウンサンプリングのボクセルサイズ

    Returns:
        tuple:
            - downsampled_coords: ダウンサンプリングされた点群の座標 (M, 3)
            - downsampled_colors: ダウンサンプリングされた点群の色 (M, 3)
    """

    if voxel_size <= 0.0:
        raise ValueError("Voxel_size is too small.")

    min_bound = np.min(point_coords, axis=0)

    voxel_min_bound = min_bound - voxel_size * 0.5

    ref_coords = (point_coords - voxel_min_bound) / voxel_size
    voxel_indices = np.floor(ref_coords).astype(np.int32)

    # unique_indices: 各ボクセルの座標を表す配列
    # inverse_indices: 各点がどのボクセルに属するかを示すインデックス
    # counts: 各ボクセルに属する点の数を表す配列
    unique_indices, inverse_indices, counts = np.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True
    )

    # 各ボクセルの点群の平均座標と色を計算
    sum_points = np.zeros(
        (unique_indices.shape[0], 3)
    )  # 各ボクセルの点群の座標の合計を格納する配列
    sum_colors = np.zeros(
        (unique_indices.shape[0], 3)
    )  # 各ボクセルの点群の色の合計を格納する配列
    for column in range(3):
        sum_points[:, column] = np.bincount(
            inverse_indices, weights=point_coords[:, column]
        )
        sum_colors[:, column] = np.bincount(
            inverse_indices, weights=point_colors[:, column]
        )
    downsampled_coords = sum_points / counts[:, np.newaxis]
    downsampled_colors = sum_colors / counts[:, np.newaxis]

    print(
        f"生データ {len(point_coords)} から {len(downsampled_coords)} にダウンサンプリング完了"
    )

    return downsampled_coords, downsampled_colors
