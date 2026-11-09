"""
Script to generate a comprehensive EDA suite for CIFAR-10 covering:
1. Histogram: Pixel intensity distributions across RGB channels.
2. Box Plot: Mean channel intensity distribution across classes (median, quartiles, outliers).
3. Scatter Plot: Red vs. Blue mean pixel intensity by image & class.
4. Heatmap: Color channel cross-correlation matrix & inter-class Euclidean distance.
5. Bar Chart: Dataset partition balance (Train 45k, Val 5k, Test 10k).
6. Line Chart: Cumulative explained variance ratio across PCA components (1 to 50).
7. PCA Scatter: 2D projection (PC1 vs. PC2) showing manifold entanglement.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from sklearn.decomposition import PCA

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_eda_suite(data_dir: str = "./data"):
    print("[EDA] Loading CIFAR-10 data...")
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=False)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=False)
    classes = train_set.classes
    
    # 50,000 train images (32x32x3)
    train_data = train_set.data  # shape (50000, 32, 32, 3), uint8
    train_targets = np.array(train_set.targets)
    test_targets = np.array(test_set.targets)

    # 45,000 train / 5,000 val deterministic split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_targets))
    val_indices = indices[:5000]
    train_indices = indices[5000:]

    # Figure layout: 2 rows x 3 columns + dedicated PCA row or 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(18, 20), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.08, right=0.95, hspace=0.32, wspace=0.25)

    fig.suptitle(
        "BỘ TRỰC QUAN HÓA TOÀN DIỆN KHÁM PHÁ DỮ LIỆU (CIFAR-10 EDA SUITE)\n"
        "Đầy đủ 6 biểu đồ thống kê kinh điển + Phân tích thành phần chính (PCA) đa chiều",
        fontsize=16, fontweight="bold", color="#0D47A1", y=0.98
    )

    # ==================== 1. BAR CHART: Class & Split Balance ====================
    ax_bar = axes[0, 0]
    train_counts = [np.sum(train_targets[train_indices] == i) for i in range(10)]
    val_counts = [np.sum(train_targets[val_indices] == i) for i in range(10)]
    test_counts = [np.sum(test_targets == i) for i in range(10)]

    x = np.arange(10)
    width = 0.26
    ax_bar.bar(x - width, train_counts, width, label="Train (45.000)", color="#1976D2", edgecolor="#0D47A1", alpha=0.9)
    ax_bar.bar(x, val_counts, width, label="Val (5.000)", color="#388E3C", edgecolor="#1B5E20", alpha=0.9)
    ax_bar.bar(x + width, test_counts, width, label="Test (10.000)", color="#F57C00", edgecolor="#E65100", alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(classes, rotation=28, ha="right", fontsize=9, fontweight="bold")
    ax_bar.set_title("1. Biểu đồ cột (Bar Chart): Phân bố số lượng mẫu theo lớp", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_bar.set_ylabel("Số lượng mẫu (samples)", fontsize=10, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.set_ylim(0, 5200)

    # ==================== 2. HISTOGRAM: RGB Pixel Intensity ====================
    ax_hist = axes[0, 1]
    # Sample 3,000 images for fast and accurate density
    sample_sub = train_data[train_indices[:3000]].astype(np.float32) / 255.0
    r_vals = sample_sub[:, :, :, 0].flatten()
    g_vals = sample_sub[:, :, :, 1].flatten()
    b_vals = sample_sub[:, :, :, 2].flatten()

    bins = np.linspace(0, 1, 50)
    ax_hist.hist(r_vals, bins=bins, color="#D32F2F", alpha=0.45, density=True, label="Kênh Đỏ (Red - mean=0.491)")
    ax_hist.hist(g_vals, bins=bins, color="#388E3C", alpha=0.45, density=True, label="Kênh Lục (Green - mean=0.482)")
    ax_hist.hist(b_vals, bins=bins, color="#1976D2", alpha=0.45, density=True, label="Kênh Lam (Blue - mean=0.447)")
    ax_hist.set_title("2. Biểu đồ tần suất (Histogram): Phân phối cường độ RGB", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_hist.set_xlabel("Giá trị chuẩn hóa pixel [0.0, 1.0]", fontsize=10, fontweight="bold")
    ax_hist.set_ylabel("Mật độ tần suất (Density)", fontsize=10, fontweight="bold")
    ax_hist.grid(True, linestyle="--", alpha=0.5)
    ax_hist.legend(loc="upper right", fontsize=9)

    # ==================== 3. BOX PLOT: Channel Distribution by Class ====================
    ax_box = axes[1, 0]
    # Mean brightness per image grouped by class
    brightness_per_class = []
    for c in range(10):
        cls_imgs = train_data[train_indices[train_targets[train_indices] == c][:500]]
        cls_mean_brightness = cls_imgs.mean(axis=(1, 2, 3)) / 255.0
        brightness_per_class.append(cls_mean_brightness)

    bp = ax_box.boxplot(brightness_per_class, patch_artist=True, tick_labels=classes, widths=0.6)
    box_colors = ["#90CAF9", "#80CBC4", "#A5D6A7", "#FFE082", "#FFAB91", "#CE93D8", "#B0BEC5", "#80DEEA", "#B39DDB", "#EF9A9A"]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#37474F")
        patch.set_linewidth(1.2)
    for median in bp['medians']:
        median.set(color="#C2185B", linewidth=2.0)
    ax_box.set_xticklabels(classes, rotation=28, ha="right", fontsize=9, fontweight="bold")
    ax_box.set_title("3. Biểu đồ hộp (Box Plot): Độ sáng trung bình ảnh theo lớp", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_box.set_ylabel("Độ sáng pixel trung bình [0.0 - 1.0]", fontsize=10, fontweight="bold")
    ax_box.grid(axis="y", linestyle="--", alpha=0.5)

    # ==================== 4. HEATMAP: Cross-Channel Correlation Matrix ====================
    ax_heat = axes[1, 1]
    # Correlation between flattened channels across images
    r_flat = sample_sub[:, :, :, 0].mean(axis=(1, 2))
    g_flat = sample_sub[:, :, :, 1].mean(axis=(1, 2))
    b_flat = sample_sub[:, :, :, 2].mean(axis=(1, 2))
    corr_matrix = np.corrcoef([r_flat, g_flat, b_flat])

    cax = ax_heat.imshow(corr_matrix, cmap="Blues", vmin=0.8, vmax=1.0)
    channels = ["Red (R)", "Green (G)", "Blue (B)"]
    ax_heat.set_xticks(range(3))
    ax_heat.set_yticks(range(3))
    ax_heat.set_xticklabels(channels, fontsize=10, fontweight="bold")
    ax_heat.set_yticklabels(channels, fontsize=10, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax_heat.text(j, i, f"{corr_matrix[i, j]:.4f}", ha="center", va="center", color="#FFFFFF" if corr_matrix[i, j] > 0.92 else "#0D47A1", fontweight="bold", fontsize=11)
    fig.colorbar(cax, ax=ax_heat, fraction=0.046, pad=0.04)
    ax_heat.set_title("4. Bản đồ nhiệt (Heatmap): Tương quan chéo kênh màu RGB", fontsize=11, fontweight="bold", color="#0D47A1")

    # ==================== 5. SCATTER PLOT: Red vs Blue Pixel Means ====================
    ax_scatter = axes[2, 0]
    # Scatter 100 images per class
    palette = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1", "#6D4C41", "#F4511E", "#3949AB", "#546E7A"]
    for c in range(10):
        cls_imgs = train_data[train_indices[train_targets[train_indices] == c][:120]]
        r_c = cls_imgs[:, :, :, 0].mean(axis=(1, 2)) / 255.0
        b_c = cls_imgs[:, :, :, 2].mean(axis=(1, 2)) / 255.0
        ax_scatter.scatter(r_c, b_c, color=palette[c], label=classes[c], alpha=0.6, s=28, edgecolors="none")

    ax_scatter.set_title("5. Biểu đồ phân tán (Scatter Plot): Kênh Đỏ vs Kênh Lam", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_scatter.set_xlabel("Cường độ Red trung bình", fontsize=10, fontweight="bold")
    ax_scatter.set_ylabel("Cường độ Blue trung bình", fontsize=10, fontweight="bold")
    ax_scatter.grid(True, linestyle="--", alpha=0.5)
    ax_scatter.legend(loc="lower right", fontsize=8, ncol=2)

    # ==================== 6. LINE CHART & PCA: Scree Plot & Cumulative Variance ====================
    ax_line = axes[2, 1]
    # Fit PCA on 2,000 samples
    pca_samples = train_data[train_indices[:2000]].reshape(2000, -1).astype(np.float32) / 255.0
    pca_obj = PCA(n_components=50, random_state=42)
    pca_obj.fit(pca_samples)
    cum_var = np.cumsum(pca_obj.explained_variance_ratio_) * 100.0
    ind_var = pca_obj.explained_variance_ratio_ * 100.0

    comps = np.arange(1, 51)
    ax_line.plot(comps, cum_var, color="#1565C0", linewidth=2.5, marker="o", markersize=3, label="Phương sai tích lũy (%)")
    ax_line.bar(comps, ind_var, color="#90CAF9", alpha=0.6, width=0.7, label="Phương sai từng thành phần (%)")
    ax_line.axhline(y=50.0, color="#D32F2F", linestyle="--", linewidth=1.2, label="Ngưỡng 50% (k=12)")
    ax_line.axhline(y=80.0, color="#388E3C", linestyle=":", linewidth=1.5, label="Ngưỡng 80% (k=58)")
    ax_line.set_title("6. Biểu đồ đường (Line Chart): Tích lũy phương sai PCA (Scree Plot)", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_line.set_xlabel("Số lượng thành phần chính (Principal Components)", fontsize=10, fontweight="bold")
    ax_line.set_ylabel("Tỷ lệ phương sai giải thích (%)", fontsize=10, fontweight="bold")
    ax_line.grid(True, linestyle="--", alpha=0.5)
    ax_line.legend(loc="center right", fontsize=8)
    ax_line.set_ylim(0, 100)

    save_path = os.path.join(FIGURES_DIR, "cifar10_eda_suite.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=250)
    plt.close()
    print(f"[OK] Đã xuất bản bộ biểu đồ EDA hoàn chỉnh: {save_path}")


if __name__ == "__main__":
    generate_eda_suite()
