import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open3Dを用いた点群レジストレーションのプロトタイプ"
    )

    # パラメータ
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="ダウンサンプリングのボクセルサイズ",
    )
    parser.add_argument(
        "--max_iter", type=int, default=30, help="ICPの最大イテレーション数"
    )
    parser.add_argument(
        "--eval_max_distance", type=float, default=0.1, help="RMSE評価時の最大対応距離"
    )
    parser.add_argument("--skip", type=int, default=5, help="描画する点の間隔")

    return parser.parse_args()


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


def draw_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    skip: int,
) -> None:
    """2つの点群を重ねて描画する。

    Args:
        source: 点群データと色データを持つ点群オブジェクト
        target: 点群データと色データを持つ点群オブジェクト
        transformation: sourceをtargetに重ねるための変換行列

    """

    source_points = np.asarray(source.points)
    source_colors = np.asarray(source.colors)

    target_points = np.asarray(target.points)
    target_colors = np.asarray(target.colors)

    R = transformation[:3, :3]
    t = transformation[:3, 3]
    transformed_source_points = np.dot(source_points, R.T) + t

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.axis("off")

    # Target点群の描写
    ax.scatter(
        target_points[::skip, 0],
        target_points[::skip, 1],
        target_points[::skip, 2],
        c=target_colors[::skip],
        s=0.5,
        alpha=0.6,
        marker=".",
        label="Target",
    )

    # 変換後Source点群の描画
    ax.scatter(
        transformed_source_points[::skip, 0],
        transformed_source_points[::skip, 1],
        transformed_source_points[::skip, 2],
        c=source_colors[::skip],
        s=0.5,
        alpha=0.6,
        marker=".",
        label="Source (Aligned)",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Registration Result")

    # 3D空間のスケール（アスペクト比）を等倍に揃えて歪みを防ぐ
    all_pts = np.vstack([target_points[::skip], transformed_source_points[::skip]])
    max_range = (
        np.array(
            [
                all_pts[:, 0].max() - all_pts[:, 0].min(),
                all_pts[:, 1].max() - all_pts[:, 1].min(),
                all_pts[:, 2].max() - all_pts[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) * 0.5
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) * 0.5
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    output_dir = Path("result")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "open3d_registration_result.png"

    plt.savefig(output_path)
    plt.close()

    print(f"位置合わせ結果の画像を保存しました: {output_path}")


def preprocess_point_cloud(source: o3d.geometry.PointCloud, voxel_size: float) -> tuple[
    o3d.geometry.PointCloud,
    o3d.pipelines.registration.Feature,
]:
    """点群のダウンサンプリングと特徴量計算を行う。

    Args:
        source: 前処理を行う点群オブジェクト
        voxel_size: ダウンサンプリングのボクセルサイズ

    Returns:
        tuple:
            - source_down: ダウンサンプリングされた点群オブジェクト
            - feature: 計算されたFPFH特徴量オブジェクト
    """
    source_down = source.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return source_down, feature


def execute_global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_feature: o3d.pipelines.registration.Feature,
    target_feature: o3d.pipelines.registration.Feature,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """RANSACアルゴリズムを用いたグローバルレジストレーションを行う。

    Args:
        source_down: 前処理済みのsource点群
        target_down: 前処理済みのtarget点群
        source_feature: source点群のFPFH特徴量
        target_feature: target点群のFPFH特徴量
        voxel_size: 特徴量計算で使用したボクセルサイズ

    Returns: RANSACによる位置合わせ結果オブジェクト
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_feature,
        target_feature,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    result_ransac: o3d.pipelines.registration.RegistrationResult,
    voxel_size: float,
    max_iter: int = 30,
    eval_max_distance: float = 0.05,
) -> o3d.pipelines.registration.RegistrationResult:
    """Point-to-plane ICPアルゴリズムを用いた局所レジストレーションを行う。

    各イテレーションのRMSEの推移をグラフとして保存する。

    Args:
        source: 元のソース点群オブジェクト。
        target: 元のターゲット点群オブジェクト。
        result_ransac: RANSACによる初期位置合わせ結果。
        voxel_size: ICPの探索距離の基準となるボクセルサイズ。
        max_iter: ICPの最大イテレーション回数。デフォルトは30。
        eval_max_distance: 各ステップの評価における最大対応距離。デフォルトは0.05。

    Returns:
        ICPによる精密な位置合わせ結果オブジェクト。
    """

    radius_normal = voxel_size * 2
    if not source.has_normals():
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
    distance_threshold = voxel_size * 0.4
    rmse_history = []

    current_transformation = result_ransac.transformation

    for i in range(1, max_iter + 1):
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=i),
        )

        current_transformation = result.transformation

        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, eval_max_distance, current_transformation
        )
        rmse_history.append(evaluation.inlier_rmse)

    output_dir = Path("result")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "icp_Convergence.png"

    plt.plot(range(1, max_iter + 1), rmse_history)
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("open3d ICP Convergence")
    plt.grid(True)

    plt.savefig(output_path)
    plt.close()

    print(f"ICPの収束グラフを保存しました: {output_path}")

    return result


if __name__ == "__main__":
    args = parse_args()

    download_data()

    pcd1_path = "data/bunny_source.txt"
    pcd2_path = "data/bunny_target.txt"

    source_raw = np.loadtxt(pcd1_path, delimiter=" ")
    target_raw = np.loadtxt(pcd2_path, delimiter=" ")

    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(source_raw[:, :3])
    target.points = o3d.utility.Vector3dVector(target_raw[:, :3])

    source.colors = o3d.utility.Vector3dVector(source_raw[:, 3:6] / 255.0)

    target.colors = o3d.utility.Vector3dVector(target_raw[:, 3:6] / 255.0)

    initial_transformation = np.identity(4)

    # 点群の前処理
    voxel_size = args.voxel_size
    source_down, source_feature = preprocess_point_cloud(source, voxel_size)
    target_down, target_feature = preprocess_point_cloud(target, voxel_size)

    # RANSACによる初期位置推定
    result_ransac = execute_global_registration(
        source_down, target_down, source_feature, target_feature, voxel_size
    )
    draw_registration_result(source, target, result_ransac.transformation, args.skip)

    # 最終評価
    result_icp = refine_registration(
        source,
        target,
        result_ransac,
        voxel_size,
        max_iter=args.max_iter,
        eval_max_distance=args.eval_max_distance,
    )
    print(f"RMSE: {result_icp.inlier_rmse:.6f}")
