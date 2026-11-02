"""
Script to generate the Experiment Monitoring Architecture Diagram for Lab 2.
Generates: outputs/figures/monitoring_architecture.png and monitors_diagram.png
Visualizes:
- Training & Validation Engine
- TensorBoardLogger Streams (Scalars, Histograms, Images, Graph)
- Disk Storage Structure (runs/<experiment>/)
- TensorBoard Live Interactive Dashboard
"""

import os
import sys
import shutil

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_monitoring_diagram():
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=240)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.axis("off")

    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.6)

    # Header
    ax.text(0.8, 9.25, "KIẾN TRÚC GIÁM SÁT THỰC NGHIỆM THỜI GIAN THỰC (MONITORING ARCHITECTURE)",
            fontsize=17, fontweight="bold", color="#0D47A1", va="top")
    ax.text(0.8, 8.85, "Hệ thống logging đa luồng qua TensorBoard: Giám sát Loss, Accuracy, Phân phối Trọng số, Ảnh Augmentation & Đồ thị Tính toán",
            fontsize=11, fontstyle="italic", color="#546E7A", va="top")
    ax.axhline(8.55, xmin=0.04, xmax=0.96, color="#CFD8DC", linewidth=1.5)

    # ========================== CỘT 1: NGUỒN PHÁT DỮ LIỆU (TRAINING ENGINE) ==========================
    # Ends at 0.7 + 3.4 = 4.1 (with pad=0.08 -> 4.18)
    engine_card = patches.FancyBboxPatch((0.7, 1.15), 3.4, 7.1, boxstyle="round,pad=0.08",
                                         facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=2.0)
    ax.add_patch(engine_card)
    ax.text(2.4, 8.0, "TIẾN TRÌNH HUẤN LUYỆN\n(TRAINING ENGINE)", ha="center", va="top",
            fontsize=11.5, fontweight="bold", color="#0D47A1", linespacing=1.2)
    ax.text(2.4, 7.42, "src/lab2/train.py", ha="center", va="top",
            fontsize=9.2, fontweight="bold", color="#1565C0")

    sub_blocks = [
        {"title": "1. Khởi tạo & Dữ liệu", "lines": ["• DataLoader (Batch 16/32)", "• Model (ResNet18 / DenseNet)", "• ImageNet Weights V1"], "bg": "#FFFFFF", "border": "#90CAF9"},
        {"title": "2. Forward & Loss", "lines": ["• CrossEntropyLoss", "• Kiểm tra NaN / Inf Loss", "• Log Top-1 Accuracy"], "bg": "#FFFFFF", "border": "#90CAF9"},
        {"title": "3. Backward & Step", "lines": ["• Adam Optimizer", "• Differential LR (1e-4 / 1e-3)", "• Trích xuất Gradients & Weights"], "bg": "#FFFFFF", "border": "#90CAF9"},
        {"title": "4. Validation Loop", "lines": ["• model.eval() + no_grad()", "• val_loss & val_acc", "• Trigger Best Checkpoint"], "bg": "#FFFFFF", "border": "#90CAF9"}
    ]

    for idx, sb in enumerate(sub_blocks):
        by = 5.95 - idx * 1.45
        box = patches.FancyBboxPatch((0.85, by), 3.1, 1.25, boxstyle="round,pad=0.06",
                                     facecolor=sb["bg"], edgecolor=sb["border"], linewidth=1.4)
        ax.add_patch(box)
        ax.text(1.0, by + 1.12, sb["title"], fontsize=9.2, fontweight="bold", color="#0D47A1", va="top")
        ly = by + 0.84
        for line in sb["lines"]:
            ax.text(1.0, ly, line, fontsize=8.0, color="#37474F", va="top")
            ly -= 0.25

    # Arrow 1: Engine -> TensorBoard Logger
    # Gap from 4.18 to 5.22. Center is 4.70.
    ax.annotate("", xy=(5.2, 4.5), xytext=(4.3, 4.5),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#1565C0", lw=2.2))
    ax.text(4.75, 4.72, "Telemetry\nStream", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#1565C0")

    # ========================== CỘT 2: MODULE LOGGING (TENSORBOARD LOGGER) ==========================
    # Starts at 5.3, width 5.4 -> ends at 10.7 (with pad=0.08 -> 5.22 to 10.78)
    logger_card = patches.FancyBboxPatch((5.3, 1.15), 5.4, 7.1, boxstyle="round,pad=0.08",
                                         facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=2.0)
    ax.add_patch(logger_card)
    ax.text(8.0, 8.0, "TENSORBOARD LOGGER MODULE", ha="center", va="top",
            fontsize=11.5, fontweight="bold", color="#4A148C")
    ax.text(8.0, 7.55, "src/lab2/tensorboard_logger.py (SummaryWriter)", ha="center", va="top",
            fontsize=9.2, fontweight="bold", color="#6A1B9A")

    streams = [
        {
            "tag": "1. SCALARS MONITOR (Đường cong độ đo)",
            "desc": "• Loss/train & Loss/val theo từng epoch\n• Accuracy/train & Accuracy/val (%)\n• Thời gian thực thi mỗi epoch (seconds)\n-> Phát hiện Overfitting / Underfitting sớm",
            "bg": "#E8F5E9", "border": "#388E3C", "c_tag": "#1B5E20"
        },
        {
            "tag": "2. HISTOGRAMS MONITOR (Phân phối trọng số)",
            "desc": "• Phân phối Weights & Biases theo từng tầng (conv1, layer1..4, fc)\n• Kiểm tra gradient vanishing / exploding\n• Xác thực requires_grad=False khóa đúng trọng số",
            "bg": "#FFF3E0", "border": "#F57C00", "c_tag": "#E65100"
        },
        {
            "tag": "3. IMAGE BATCH MONITOR (Trực quan hóa ảnh)",
            "desc": "• Ảnh sau tiền xử lý (Resize 128x128)\n• Kiểm chứng RandomCrop (padding reflect) & RandomHorizontalFlip\n• Đảm bảo dữ liệu đầu vào mô hình không bị dị dạng",
            "bg": "#E0F7FA", "border": "#00838F", "c_tag": "#006064"
        },
        {
            "tag": "4. GRAPH MONITOR (Đồ thị kiến trúc tính toán)",
            "desc": "• PyTorch JIT execution graph (torch.jit.trace)\n• Trực quan hóa luồng tensor qua Residual / Dense blocks\n• Kiểm chứng số kênh đầu ra Classifier Head = 10",
            "bg": "#EDE7F6", "border": "#512DA8", "c_tag": "#311B92"
        }
    ]

    for idx, st in enumerate(streams):
        sy = 5.95 - idx * 1.45
        box = patches.FancyBboxPatch((5.5, sy), 5.0, 1.25, boxstyle="round,pad=0.06",
                                     facecolor=st["bg"], edgecolor=st["border"], linewidth=1.5)
        ax.add_patch(box)
        ax.text(5.65, sy + 1.12, st["tag"], fontsize=9.0, fontweight="bold", color=st["c_tag"], va="top")
        ax.text(5.65, sy + 0.86, st["desc"], fontsize=7.8, color="#212121", linespacing=1.3, va="top")

    # Arrow 2: Logger -> Storage & Dashboard
    # Gap from 10.78 to 11.82. Center is 11.30.
    ax.annotate("", xy=(11.8, 4.5), xytext=(10.9, 4.5),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#7B1FA2", lw=2.2))
    ax.text(11.35, 4.72, "Atomic\nWrite", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#7B1FA2")

    # ========================== CỘT 3: LƯU TRỮ VÀ DASHBOARD ==========================
    # Starts at 11.9, width 3.6 -> ends at 15.5 (with pad=0.08 -> 11.82 to 15.58)
    dash_card = patches.FancyBboxPatch((11.9, 1.15), 3.6, 7.1, boxstyle="round,pad=0.08",
                                       facecolor="#ECEFF1", edgecolor="#455A64", linewidth=2.0)
    ax.add_patch(dash_card)
    ax.text(13.7, 8.0, "LƯU TRỮ & GIAO DIỆN\n(STORAGE & DASHBOARD)", ha="center", va="top",
            fontsize=11.5, fontweight="bold", color="#263238", linespacing=1.2)
    ax.text(13.7, 7.42, "Local Disk & Web UI (Port 6006)", ha="center", va="top",
            fontsize=9.2, fontweight="bold", color="#37474F")

    # Storage Box
    store_box = patches.FancyBboxPatch((12.05, 4.45), 3.3, 2.75, boxstyle="round,pad=0.06",
                                       facecolor="#FFFFFF", edgecolor="#78909C", linewidth=1.5)
    ax.add_patch(store_box)
    ax.text(12.2, 7.05, "CẤU TRÚC LOG TRÊN ĐĨA:", fontsize=9.2, fontweight="bold", color="#37474F", va="top")
    ax.text(12.2, 6.70,
            "runs/\n"
            "├── resnet18_transfer/\n"
            "│   └── events.out.tfevents.*\n"
            "├── resnet18_finetune/\n"
            "├── densenet121_transfer/\n"
            "├── densenet121_finetune/\n"
            "├── resnet18_transfer_lr1e4/\n"
            "└── resnet18_transfer_bs32/",
            fontsize=8.0, fontfamily="monospace", color="#263238", linespacing=1.2, va="top")

    # UI Dashboard Box
    ui_box = patches.FancyBboxPatch((12.05, 1.35), 3.3, 2.85, boxstyle="round,pad=0.06",
                                    facecolor="#FFF9C4", edgecolor="#FBC02D", linewidth=1.6)
    ax.add_patch(ui_box)
    ax.text(12.2, 4.05, "TENSORBOARD DASHBOARD", fontsize=9.2, fontweight="bold", color="#F57F17", va="top")
    ax.text(12.2, 3.68,
            "• Khởi chạy trực tiếp bằng lệnh:\n"
            "  tensorboard --logdir=runs\n\n"
            "• Giao diện trực quan trên trình duyệt:\n"
            "  http://localhost:6006\n\n"
            "• So sánh song song 6 thí nghiệm\n"
            "• Smoothing loss, tải dữ liệu CSV/SVG",
            fontsize=8.0, color="#37474F", linespacing=1.25, va="top")

    # Footer note
    ax.text(0.7, 0.75,
            "* Nguyên tắc triển khai: Module TensorBoardLogger tự động đóng luồng an toàn qua khối try...finally logger.close(), "
            "đảm bảo không bị rò rỉ bộ nhớ hay khóa file sự kiện (events file) trên hệ thống.",
            fontsize=9.0, fontstyle="italic", color="#546E7A", va="top")

    save_path = os.path.join(FIGURES_DIR, "monitoring_architecture.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Also update monitors_diagram.png alias
    alias_path = os.path.join(FIGURES_DIR, "monitors_diagram.png")
    shutil.copyfile(save_path, alias_path)
    print(f"[OK] Đã lưu sơ đồ giám sát (Monitoring Diagram): {save_path}")
    print(f"[OK] Đã đồng bộ alias: {alias_path}")


if __name__ == "__main__":
    generate_monitoring_diagram()
