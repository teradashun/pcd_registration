# グローバルレジストレーション後の点群画像を表示

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import open3d as o3d

from preprocessing import downsample_point_cloud
from utils import search_hybrid, estimate_normals
from icp import register_centroids
from draw import draw_registration_result
from FPFHFeature import compute_fpfh
from ransac import find_feature_matches, ransac_feature_match


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


def parse_args():
    parser = argparse.ArgumentParser(description="Point Cloud Registration using ICP")

    # 手法の選択
    parser.add_argument(
        "--method",
        type=str,
        choices=["none", "centroid", "fpfh"],
        default="centroid",
        help="グローバルレジストレーションの方法を選択 (none, centroid, fpfh)",
    )

    # 前処理パラメータ
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.005,
        help="ダウンサンプリングのボクセルサイズ",
    )

    # RANSAC パラメータ
    parser.add_argument(
        "--ransac_iter", type=int, default=10000, help="RANSACのイテレーション数"
    )
    parser.add_argument(
        "--ransac_threshold", type=float, default=0.05, help="RANSACのインライア判定閾値"
    )

    # 法線・FPFH パラメータ
    parser.add_argument(
        "--normal_radius", type=float, default=0.02, help="法線推定用の探索半径"
    )
    parser.add_argument(
        "--normal_neighbors",
        type=int,
        default=10,
        help="法線推定に用いる近傍点の最大数",
    )

    # 描画パラメータ
    parser.add_argument("--skip", type=int, default=1, help="描画する点の間隔")

    return parser.parse_args()


args = parse_args()

download_data()

source_pcd_path = "data/bunny_source.txt"
target_pcd_path = "data/bunny_target.txt"

source_raw = np.loadtxt(source_pcd_path, delimiter=" ")
target_raw = np.loadtxt(target_pcd_path, delimiter=" ")

source_down, _ = downsample_point_cloud(
    source_raw[:, :3], source_raw[:, 3:6], voxel_size=args.voxel_size
)
target_down, _ = downsample_point_cloud(
    target_raw[:, :3], target_raw[:, 3:6], voxel_size=args.voxel_size
)

target_normals = None
if args.method == "fpfh":
    print("Target点群の法線を推定中...")
    target_neighbors = search_hybrid(
        data=target_down,
        query=target_down,
        radius=args.normal_radius,
        max_neighbors=args.normal_neighbors,
    )
    target_normals = estimate_normals(target_down, target_neighbors)
    print("法線推定完了")
    print("Source点群の法線を推定中")
    source_neighbors = search_hybrid(
        data=source_down,
        query=source_down,
        radius=args.normal_radius,
        max_neighbors=args.normal_neighbors,
    )
    source_normals = estimate_normals(source_down, source_neighbors)

    print("FPFH特徴量の計算を開始")
    target_fpfh = compute_fpfh(target_down, target_normals, target_neighbors)
    source_fpfh = compute_fpfh(source_down, source_normals, source_neighbors)

    print("RANSACによるグローバルレジストレーションを実行中...")
    matches = find_feature_matches(source_fpfh, target_fpfh)
    T_init = ransac_feature_match(
        source_down,
        target_down,
        matches,
        max_iterations=args.ransac_iter,
        inlier_threshold=args.ransac_threshold,
    )

elif args.method == "centroid":
    print("重心合わせによる初期化を実行中...")
    T_init = register_centroids(source_down, target_down)

else:
    print("生の点群画像を表示")
    T_init = np.identity(4)

draw_registration_result(source_raw, target_raw, T_init, args.skip, args.method)
