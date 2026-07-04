import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def draw_registration_result(
    source: np.ndarray,
    target: np.ndarray,
    transformation: np.ndarray,
    skip: int,
    method: str,
) -> None:
    """2つの点群を重ねて描画し、画像として保存する。

    Args:
        source: 元のsource点群配列 (N, 6)
        target: 元のtarget点群配列 (M, 6)
        transformation: 累積された4x4の変換行列
        skip: 描画する点の間隔
        method: 使用したICP手法の名前
    """

    source_points = source[:, :3]
    source_colors = source[:, 3:6] / 255.0

    target_points = target[:, :3]
    target_colors = target[:, 3:6] / 255.0

    R = transformation[:3, :3]
    t = transformation[:3, 3]
    transformed_source_points = np.dot(source_points, R.T) + t

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.axis("off")

    # target点群の描写
    ax.scatter(
        target_points[::skip, 0],
        target_points[::skip, 1],
        target_points[::skip, 2],
        c=target_colors[::skip],
        s=0.5,
        alpha=0.5,
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
        alpha=0.5,
        marker=".",
        label="Source (Aligned)",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Registration Result")

    # 3D空間のスケール（アスペクト比）を等倍に揃えて歪みを防ぐ
    all_points = np.vstack([target_points[::skip], transformed_source_points[::skip]])
    max_range = (
        np.array(
            [
                all_points[:, 0].max() - all_points[:, 0].min(),
                all_points[:, 1].max() - all_points[:, 1].min(),
                all_points[:, 2].max() - all_points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    output_dir = Path("result")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"registration_{method}.png"

    plt.savefig(output_path)
    plt.close()

    print(f"位置合わせ結果の画像を保存しました: {output_path}")
