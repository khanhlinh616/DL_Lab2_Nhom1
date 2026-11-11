"""
Script to generate a comprehensive EDA suite for CIFAR-10 covering:
1. Histogram: Pixel intensity distributions across RGB channels.
2. Box Plot: Mean channel intensity distribution across classes (median, quartiles, outliers).
3. Scatter Plot: Red vs. Blue mean pixel intensity by image & class.
4. Heatmap: Color channel cross-correlation matrix & inter-class Euclidean distance.
5. Bar Chart: Dataset partition balance (Train 45k, Val 5k, Test 10k).
6. Line Chart: Cumulative explained variance ratio across PCA components (1 to 50).
7. PCA Scatter: 2D projection (PC1 vs. PC2) showing manifold entanglement.

Saves BOTH individual standalone charts (cifar10_eda_*.png) and composite image.
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

    # Sample 3,000 images for fast & accurate distributions
    sample_sub = train_data[train_indices[:3000]].astype(np.float32) / 255.0
    r_vals = sample_sub[:, :, :, 0].flatten()
    g_vals = sample_sub[:, :, :, 1].flatten()
    b_vals = sample_sub[:, :, :, 2].flatten()

    # Precalculate data for each chart
    train_counts = [np.sum(train_targets[train_indices] == i) for i in range(10)]
    val_counts = [np.sum(train_targets[val_indices] == i) for i in range(10)]
    test_counts = [np.sum(test_targets == i) for i in range(10)]

    # ==================== 1. INDIVIDUAL: BAR CHART ====================
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5.5), dpi=220)
    fig_bar.patch.set_facecolor("#FFFFFF")
    x = np.arange(10)
    width = 0.26
    ax_bar.bar(x - width, train_counts, width, label="Train (45.000)", color="#1976D2", edgecolor="#0D47A1", alpha=0.9)
    ax_bar.bar(x, val_counts, width, label="Val (5.000)", color="#388E3C", edgecolor="#1B5E20", alpha=0.9)
    ax_bar.bar(x + width, test_counts, width, label="Test (10.000)", color="#F57C00", edgecolor="#E65100", alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(classes, rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax_bar.set_title("Biểu đồ cột: Phân bố số lượng mẫu theo 10 lớp và phân vùng (Train/Val/Test)", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_bar.set_ylabel("Số lượng mẫu (samples)", fontsize=10, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
    ax_bar.legend(loc="upper right", fontsize=10)
    ax_bar.set_ylim(0, 5300)
    bar_path = os.path.join(FIGURES_DIR, "cifar10_eda_1_bar_distribution.png")
    fig_bar.savefig(bar_path, bbox_inches="tight", dpi=250)
    plt.close(fig_bar)

    # ==================== 2. INDIVIDUAL: HISTOGRAM ====================
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5.5), dpi=220)
    fig_hist.patch.set_facecolor("#FFFFFF")
    bins = np.linspace(0, 1, 50)
    ax_hist.hist(r_vals, bins=bins, color="#D32F2F", alpha=0.45, density=True, label="Kênh Đỏ (Red - mean=0.491)")
    ax_hist.hist(g_vals, bins=bins, color="#388E3C", alpha=0.45, density=True, label="Kênh Lục (Green - mean=0.482)")
    ax_hist.hist(b_vals, bins=bins, color="#1976D2", alpha=0.45, density=True, label="Kênh Lam (Blue - mean=0.447)")
    ax_hist.set_title("Biểu đồ tần suất: Phân phối hàm mật độ xác suất cường độ pixel RGB", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_hist.set_xlabel("Giá trị chuẩn hóa pixel [0.0, 1.0]", fontsize=10, fontweight="bold")
    ax_hist.set_ylabel("Mật độ tần suất (Density)", fontsize=10, fontweight="bold")
    ax_hist.grid(True, linestyle="--", alpha=0.5)
    ax_hist.legend(loc="upper right", fontsize=10)
    hist_path = os.path.join(FIGURES_DIR, "cifar10_eda_2_histogram_rgb.png")
    fig_hist.savefig(hist_path, bbox_inches="tight", dpi=250)
    plt.close(fig_hist)

    # ==================== 3. INDIVIDUAL: BOX PLOT ====================
    fig_box, ax_box = plt.subplots(figsize=(10, 5.5), dpi=220)
    fig_box.patch.set_facecolor("#FFFFFF")
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
    ax_box.set_xticklabels(classes, rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax_box.set_title("Biểu đồ hộp: Độ sáng trung bình của ảnh theo từng lớp đối tượng (Median, Q1-Q3 & Outliers)", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_box.set_ylabel("Độ sáng pixel trung bình [0.0 - 1.0]", fontsize=10, fontweight="bold")
    ax_box.grid(axis="y", linestyle="--", alpha=0.5)
    box_path = os.path.join(FIGURES_DIR, "cifar10_eda_3_boxplot_brightness.png")
    fig_box.savefig(box_path, bbox_inches="tight", dpi=250)
    plt.close(fig_box)

    # ==================== 4. INDIVIDUAL: HEATMAP ====================
    fig_heat, ax_heat = plt.subplots(figsize=(8, 6.2), dpi=220)
    fig_heat.patch.set_facecolor("#FFFFFF")
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
            ax_heat.text(j, i, f"{corr_matrix[i, j]:.4f}", ha="center", va="center", color="#FFFFFF" if corr_matrix[i, j] > 0.92 else "#0D47A1", fontweight="bold", fontsize=12)
    fig_heat.colorbar(cax, ax=ax_heat, fraction=0.046, pad=0.04)
    ax_heat.set_title("Bản đồ nhiệt: Ma trận tương quan chéo giữa các kênh quang phổ RGB", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    heat_path = os.path.join(FIGURES_DIR, "cifar10_eda_4_heatmap_correlation.png")
    fig_heat.savefig(heat_path, bbox_inches="tight", dpi=250)
    plt.close(fig_heat)

    # ==================== 5. INDIVIDUAL: SCATTER PLOT ====================
    fig_scat, ax_scatter = plt.subplots(figsize=(10, 6.0), dpi=220)
    fig_scat.patch.set_facecolor("#FFFFFF")
    palette = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1", "#6D4C41", "#F4511E", "#3949AB", "#546E7A"]
    for c in range(10):
        cls_imgs = train_data[train_indices[train_targets[train_indices] == c][:120]]
        r_c = cls_imgs[:, :, :, 0].mean(axis=(1, 2)) / 255.0
        b_c = cls_imgs[:, :, :, 2].mean(axis=(1, 2)) / 255.0
        ax_scatter.scatter(r_c, b_c, color=palette[c], label=classes[c], alpha=0.6, s=32, edgecolors="none")

    ax_scatter.set_title("Biểu đồ phân tán: Tương quan cường độ kênh Red vs. Kênh Blue theo lớp", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_scatter.set_xlabel("Cường độ Red trung bình", fontsize=10, fontweight="bold")
    ax_scatter.set_ylabel("Cường độ Blue trung bình", fontsize=10, fontweight="bold")
    ax_scatter.grid(True, linestyle="--", alpha=0.5)
    ax_scatter.legend(loc="lower right", fontsize=9, ncol=2)
    scat_path = os.path.join(FIGURES_DIR, "cifar10_eda_5_scatter_rgb.png")
    fig_scat.savefig(scat_path, bbox_inches="tight", dpi=250)
    plt.close(fig_scat)

    # ==================== 6. INDIVIDUAL: LINE CHART (SCREE PLOT) ====================
    fig_line, ax_line = plt.subplots(figsize=(10, 5.5), dpi=220)
    fig_line.patch.set_facecolor("#FFFFFF")
    pca_samples = train_data[train_indices[:2000]].reshape(2000, -1).astype(np.float32) / 255.0
    pca_obj = PCA(n_components=50, random_state=42)
    X_pca_2d = pca_obj.fit_transform(pca_samples)
    cum_var = np.cumsum(pca_obj.explained_variance_ratio_) * 100.0
    ind_var = pca_obj.explained_variance_ratio_ * 100.0

    comps = np.arange(1, 51)
    ax_line.plot(comps, cum_var, color="#1565C0", linewidth=2.5, marker="o", markersize=3, label="Phương sai tích lũy (%)")
    ax_line.bar(comps, ind_var, color="#90CAF9", alpha=0.6, width=0.7, label="Phương sai từng thành phần (%)")
    ax_line.axhline(y=50.0, color="#D32F2F", linestyle="--", linewidth=1.2, label="Ngưỡng 50% (k=12)")
    ax_line.axhline(y=80.0, color="#388E3C", linestyle=":", linewidth=1.5, label="Ngưỡng 80% (k=58)")
    ax_line.set_title("Biểu đồ đường (Scree Plot): Tỷ lệ phương sai giải thích tích lũy qua 50 thành phần PCA", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_line.set_xlabel("Số lượng thành phần chính (Principal Components)", fontsize=10, fontweight="bold")
    ax_line.set_ylabel("Tỷ lệ phương sai giải thích (%)", fontsize=10, fontweight="bold")
    ax_line.grid(True, linestyle="--", alpha=0.5)
    ax_line.legend(loc="center right", fontsize=9)
    ax_line.set_ylim(0, 100)
    line_path = os.path.join(FIGURES_DIR, "cifar10_eda_6_line_pca_scree.png")
    fig_line.savefig(line_path, bbox_inches="tight", dpi=250)
    plt.close(fig_line)

    # ==================== 7. INDIVIDUAL: PCA 2D SCATTER PLOT ====================
    fig_pca2d, ax_pca = plt.subplots(figsize=(11, 7.5), dpi=220)
    fig_pca2d.patch.set_facecolor("#FFFFFF")
    ax_pca.set_facecolor("#FAFAFA")
    y_sub = train_targets[train_indices[:2000]]
    for c_idx in range(10):
        mask = (y_sub == c_idx)
        ax_pca.scatter(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            c=palette[c_idx], label=f"{c_idx}: {classes[c_idx]}",
            alpha=0.5, s=26, edgecolors="none"
        )
        cx = np.mean(X_pca_2d[mask, 0])
        cy = np.mean(X_pca_2d[mask, 1])
        ax_pca.scatter(cx, cy, color=palette[c_idx], s=160, edgecolors="#0D47A1", linewidth=2.0, zorder=6)
        ax_pca.text(cx, cy + 0.4, classes[c_idx], fontsize=9, fontweight="bold", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=palette[c_idx], alpha=0.9))

    ax_pca.set_title("Biểu đồ giảm chiều PCA 2D (PC1 vs. PC2): Hiện tượng chồng lấn đa tạp (Manifold Entanglement)", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_pca.set_xlabel(f"PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}% phương sai) — Độ sáng tổng thể", fontsize=10, fontweight="bold")
    ax_pca.set_ylabel(f"PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}% phương sai) — Phân tách dải màu đứng", fontsize=10, fontweight="bold")
    ax_pca.grid(True, linestyle="--", alpha=0.5)
    ax_pca.legend(loc="upper right", fontsize=8, ncol=2)
    pca2d_path = os.path.join(FIGURES_DIR, "cifar10_eda_7_pca_2d_scatter.png")
    fig_pca2d.savefig(pca2d_path, bbox_inches="tight", dpi=250)
    plt.close(fig_pca2d)

    # ==================== COMPOSITE 3x2 SUITE ====================
    fig, axes = plt.subplots(3, 2, figsize=(18, 20), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.08, right=0.95, hspace=0.32, wspace=0.25)
    fig.suptitle(
        "BỘ TRỰC QUAN HÓA TOÀN DIỆN KHÁM PHÁ DỮ LIỆU (CIFAR-10 EDA SUITE)\n"
        "Đầy đủ 6 biểu đồ thống kê kinh điển + Phân tích thành phần chính (PCA) đa chiều",
        fontsize=16, fontweight="bold", color="#0D47A1", y=0.98
    )

    # 1. Bar
    ax_bar2 = axes[0, 0]
    ax_bar2.bar(x - width, train_counts, width, label="Train (45.000)", color="#1976D2", edgecolor="#0D47A1", alpha=0.9)
    ax_bar2.bar(x, val_counts, width, label="Val (5.000)", color="#388E3C", edgecolor="#1B5E20", alpha=0.9)
    ax_bar2.bar(x + width, test_counts, width, label="Test (10.000)", color="#F57C00", edgecolor="#E65100", alpha=0.9)
    ax_bar2.set_xticks(x)
    ax_bar2.set_xticklabels(classes, rotation=28, ha="right", fontsize=9, fontweight="bold")
    ax_bar2.set_title("1. Biểu đồ cột (Bar Chart): Phân bố số lượng mẫu theo lớp", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_bar2.set_ylabel("Số lượng mẫu (samples)", fontsize=10, fontweight="bold")
    ax_bar2.grid(axis="y", linestyle="--", alpha=0.5)
    ax_bar2.legend(loc="upper right", fontsize=9)
    ax_bar2.set_ylim(0, 5200)

    # 2. Hist
    ax_hist2 = axes[0, 1]
    ax_hist2.hist(r_vals, bins=bins, color="#D32F2F", alpha=0.45, density=True, label="Kênh Đỏ (Red - mean=0.491)")
    ax_hist2.hist(g_vals, bins=bins, color="#388E3C", alpha=0.45, density=True, label="Kênh Lục (Green - mean=0.482)")
    ax_hist2.hist(b_vals, bins=bins, color="#1976D2", alpha=0.45, density=True, label="Kênh Lam (Blue - mean=0.447)")
    ax_hist2.set_title("2. Biểu đồ tần suất (Histogram): Phân phối cường độ RGB", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_hist2.set_xlabel("Giá trị chuẩn hóa pixel [0.0, 1.0]", fontsize=10, fontweight="bold")
    ax_hist2.set_ylabel("Mật độ tần suất (Density)", fontsize=10, fontweight="bold")
    ax_hist2.grid(True, linestyle="--", alpha=0.5)
    ax_hist2.legend(loc="upper right", fontsize=9)

    # 3. Box
    ax_box2 = axes[1, 0]
    bp2 = ax_box2.boxplot(brightness_per_class, patch_artist=True, tick_labels=classes, widths=0.6)
    for patch, color in zip(bp2['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#37474F")
        patch.set_linewidth(1.2)
    for median in bp2['medians']:
        median.set(color="#C2185B", linewidth=2.0)
    ax_box2.set_xticklabels(classes, rotation=28, ha="right", fontsize=9, fontweight="bold")
    ax_box2.set_title("3. Biểu đồ hộp (Box Plot): Độ sáng trung bình ảnh theo lớp", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_box2.set_ylabel("Độ sáng pixel trung bình [0.0 - 1.0]", fontsize=10, fontweight="bold")
    ax_box2.grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Heatmap
    ax_heat2 = axes[1, 1]
    cax2 = ax_heat2.imshow(corr_matrix, cmap="Blues", vmin=0.8, vmax=1.0)
    ax_heat2.set_xticks(range(3))
    ax_heat2.set_yticks(range(3))
    ax_heat2.set_xticklabels(channels, fontsize=10, fontweight="bold")
    ax_heat2.set_yticklabels(channels, fontsize=10, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax_heat2.text(j, i, f"{corr_matrix[i, j]:.4f}", ha="center", va="center", color="#FFFFFF" if corr_matrix[i, j] > 0.92 else "#0D47A1", fontweight="bold", fontsize=11)
    fig.colorbar(cax2, ax=ax_heat2, fraction=0.046, pad=0.04)
    ax_heat2.set_title("4. Bản đồ nhiệt (Heatmap): Tương quan chéo kênh màu RGB", fontsize=11, fontweight="bold", color="#0D47A1")

    # 5. Scatter
    ax_scatter2 = axes[2, 0]
    for c in range(10):
        cls_imgs = train_data[train_indices[train_targets[train_indices] == c][:120]]
        r_c = cls_imgs[:, :, :, 0].mean(axis=(1, 2)) / 255.0
        b_c = cls_imgs[:, :, :, 2].mean(axis=(1, 2)) / 255.0
        ax_scatter2.scatter(r_c, b_c, color=palette[c], label=classes[c], alpha=0.6, s=28, edgecolors="none")
    ax_scatter2.set_title("5. Biểu đồ phân tán (Scatter Plot): Kênh Đỏ vs Kênh Lam", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_scatter2.set_xlabel("Cường độ Red trung bình", fontsize=10, fontweight="bold")
    ax_scatter2.set_ylabel("Cường độ Blue trung bình", fontsize=10, fontweight="bold")
    ax_scatter2.grid(True, linestyle="--", alpha=0.5)
    ax_scatter2.legend(loc="lower right", fontsize=8, ncol=2)

    # 6. Line
    ax_line2 = axes[2, 1]
    ax_line2.plot(comps, cum_var, color="#1565C0", linewidth=2.5, marker="o", markersize=3, label="Phương sai tích lũy (%)")
    ax_line2.bar(comps, ind_var, color="#90CAF9", alpha=0.6, width=0.7, label="Phương sai từng thành phần (%)")
    ax_line2.axhline(y=50.0, color="#D32F2F", linestyle="--", linewidth=1.2, label="Ngưỡng 50% (k=12)")
    ax_line2.axhline(y=80.0, color="#388E3C", linestyle=":", linewidth=1.5, label="Ngưỡng 80% (k=58)")
    ax_line2.set_title("6. Biểu đồ đường (Line Chart): Tích lũy phương sai PCA (Scree Plot)", fontsize=11, fontweight="bold", color="#0D47A1")
    ax_line2.set_xlabel("Số lượng thành phần chính (Principal Components)", fontsize=10, fontweight="bold")
    ax_line2.set_ylabel("Tỷ lệ phương sai giải thích (%)", fontsize=10, fontweight="bold")
    ax_line2.grid(True, linestyle="--", alpha=0.5)
    ax_line2.legend(loc="center right", fontsize=8)
    ax_line2.set_ylim(0, 100)

    suite_path = os.path.join(FIGURES_DIR, "cifar10_eda_suite.png")
    fig.savefig(suite_path, bbox_inches="tight", dpi=250)
    plt.close(fig)

    print(f"[OK] Đã xuất bản 7 biểu đồ riêng biệt và bộ suite tổng hợp tại: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_eda_suite()
