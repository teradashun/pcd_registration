import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from preprocessing import downsample_point_cloud
from utils import (
    download_data,
    download_and_make_missing_data,
    search_hybrid,
    estimate_normals,
)
from icp import (
    register_centroids,
    compute_rmse,
    estimate_rigid_transform,
    estimate_point_to_plane_transform,
)
from draw import draw_registration_result
from FPFHFeature import compute_fpfh
from ransac import find_feature_matches, ransac_feature_match


def parse_args():
    parser = argparse.ArgumentParser(description="Point Cloud Registration using ICP")

    # 手法の選択
    parser.add_argument(
        "--method",
        type=str,
        choices=["p2p", "p2pl", "fpfh_p2p", "fpfh_p2pl"],
        default="p2p",
        help="ICPの手法を選択 (p2p, p2pl, fpfh_p2p, fpfh_p2pl)",
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
        "--ransac_threshold",
        type=float,
        default=0.05,
        help="RANSACのインライア判定閾値",
    )

    # ICPパラメータ
    parser.add_argument(
        "--max_iter", type=int, default=30, help="ICPの最大イテレーション数"
    )
    parser.add_argument(
        "--init_radius", type=float, default=0.05, help="初期RMSE計算用の探索半径"
    )
    parser.add_argument(
        "--icp_radius",
        type=float,
        default=0.02,
        help="ICPループ内での対応点探索用の探索半径",
    )
    parser.add_argument(
        "--rmse_radius",
        type=float,
        default=0.05,
        help="ICPループ内でのRMSE計算用の探索半径",
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


if __name__ == "__main__":
    args = parse_args()

    print(f"レジストレーション手法: {args.method}")

    # 欠損なし「download_data」 or 欠損あり「download_and_make_missing_data」
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
    if args.method in ["p2pl", "fpfh_p2p", "fpfh_p2pl"]:
        print("Target点群の法線を推定中...")
        target_neighbors = search_hybrid(
            data=target_down,
            query=target_down,
            radius=args.normal_radius,
            max_neighbors=args.normal_neighbors,
        )
        target_normals = estimate_normals(target_down, target_neighbors)
        print("法線推定完了")

    if args.method in ["fpfh_p2p", "fpfh_p2pl"]:
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

    # 初期位置合わせ
    if args.method in ["fpfh_p2p", "fpfh_p2pl"]:
        print("RANSACによる初期位置合わせを実行中...")
        matches = find_feature_matches(source_fpfh, target_fpfh)
        T_init = ransac_feature_match(
            source_down,
            target_down,
            matches,
            max_iterations=args.ransac_iter,
            inlier_threshold=args.ransac_threshold,
        )
    else:
        print("重心合わせによる初期化を実行中...")
        T_init = register_centroids(source_down, target_down)

    source_down_matched = np.dot(source_down, T_init[:3, :3].T) + T_init[:3, 3]
    current_source_raw = source_raw.copy()
    current_source_raw[:, :3] = (
        np.dot(source_raw[:, :3], T_init[:3, :3].T) + T_init[:3, 3]
    )

    # ICP
    rmse_history = []

    # 初期状態のRMSEを計算
    init_neighbors = search_hybrid(
        data=target_raw[:, :3],
        query=current_source_raw[:, :3],
        radius=args.init_radius,
        max_neighbors=1,
    )
    init_rmse = compute_rmse(
        current_source_raw[:, :3], target_raw[:, :3], init_neighbors
    )
    rmse_history.append(init_rmse)
    print(f"イテレーション0 : RMSE= {init_rmse:.4f}m")

    T_cumulative = T_init.copy()

    use_point2plane = args.method in ["p2pl", "fpfh_p2pl"]

    for iteration in range(1, args.max_iter + 1):
        neighbors = search_hybrid(
            data=target_down,
            query=source_down_matched,
            radius=args.icp_radius,
            max_neighbors=1,
        )

        source_paired = []
        target_paired = []
        target_normals_paired = []

        for i, neighbor_list in enumerate(neighbors):
            if len(neighbor_list) > 0:
                idx = neighbor_list[0]
                source_paired.append(source_down_matched[i])
                target_paired.append(target_down[idx])

                if use_point2plane:
                    target_normals_paired.append(target_normals[idx])

        source_paired = np.array(source_paired)
        target_paired = np.array(target_paired)

        if len(source_paired) < 3:
            print("対応点が足りないため、ICPを終了")
            break

        if use_point2plane:
            target_normals_paired = np.array(target_normals_paired)
            T_delta = estimate_point_to_plane_transform(
                source_paired, target_paired, target_normals_paired
            )
        else:
            T_delta = estimate_rigid_transform(source_paired, target_paired)

        # 点群の更新
        source_down_matched = (
            np.dot(source_down_matched, T_delta[:3, :3].T) + T_delta[:3, 3]
        )
        current_source_raw[:, :3] = (
            np.dot(current_source_raw[:, :3], T_delta[:3, :3].T) + T_delta[:3, 3]
        )

        T_cumulative = np.dot(T_delta, T_cumulative)

        raw_neighbors = search_hybrid(
            data=target_raw[:, :3],
            query=current_source_raw[:, :3],
            radius=args.rmse_radius,
            max_neighbors=1,
        )
        current_rmse = compute_rmse(
            current_source_raw[:, :3], target_raw[:, :3], raw_neighbors
        )
        rmse_history.append(current_rmse)

        print(f"イテレーション{iteration} : RMSE= {current_rmse:.4f}m")

    output_dir = Path("result")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"icp_Convergence_{args.method}.png"

    plt.plot(range(len(rmse_history)), rmse_history)
    plt.xlabel("Iteration")
    plt.ylabel("RMSE (m)")
    plt.title(f"ICP Convergence {args.method}")
    plt.grid(True)

    plt.savefig(output_path)
    plt.close()

    print(f"ICPの収束グラフを保存しました: {output_path}")

    final_rmse = rmse_history[-1]
    print(f"final_rmse: {final_rmse:.4f}m")

    draw_registration_result(
        source_raw, target_raw, T_cumulative, args.skip, args.method
    )

    # CSV出力
    csv_path = output_dir / f"RMSE_{args.method}.csv"
    try:
        csv_data = np.column_stack((np.arange(len(rmse_history)), rmse_history))
        np.savetxt(
            csv_path,
            csv_data,
            delimiter=",",
            fmt=["%d", "%.6f"],
            header="Iteration,RMSE",
            comments="",
        )
        print(f"ICPの履歴をCSVに保存しました: {csv_path}")
    except Exception as e:
        print(f"CSVの保存中にエラーが発生しました: {e}")
