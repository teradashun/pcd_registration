import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

point2point_file = "result/RMSE_point2point.csv"
point2plane_file = "result/RMSE_point2plane.csv"
fpfh_file = "result/RMSE_fpfh.csv"

files = [point2point_file, point2plane_file, fpfh_file]

labels = [
    "Point-to-Point ICP (Centroid Init)",
    "Point-to-Plane ICP (Centroid Init)",
    "Point-to-Plane ICP (RANSAC+FPFH Init)",
]

output_dir = Path("result")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "RMSE convergence.png"

plt.figure(figsize=(8, 5))

for file, label in zip(files, labels):
    data = np.loadtxt(file, delimiter=",", skiprows=1)

    x = data[:, 0]
    y = data[:, 1]

    plt.plot(x, y, label=label)

plt.xlabel("Iteration")
plt.ylabel("RMSE (m)")
plt.title("RMSE convergence")
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_locator(MultipleLocator(5))

plt.savefig(output_path)
plt.close
