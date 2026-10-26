# Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning (CIFAR-10)


Dự án được xây dựng theo chuẩn công nghiệp: **Source-First, Design-First, Test-Driven, Reproducible và CPU-First**.

---

## 1. Mục tiêu và Phạm vi

- **Bài toán:** Phân loại 10 lớp trên bộ dữ liệu ảnh **CIFAR-10** (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Mục tiêu:**
  - Nạp và sử dụng pretrained weights chính thức từ `torchvision.models`.
  - So sánh đối chứng 2 kiến trúc: **ResNet18** và **DenseNet121**.
  - So sánh 2 chiến lược thích nghi: **Transfer Learning** (đóng băng backbone, chỉ học classifier) và **Fine-Tuning** (mở khóa các khối tích chập sâu nhất và classifier).
  - Giám sát toàn bộ quá trình qua **TensorBoard**.
  - Đánh giá không thiên vị (unbiased evaluation) trên tập Test độc lập sau khi chọn Checkpoint tốt nhất qua Validation metric.
- **Ràng buộc môi trường:** Thiết kế theo hướng **CPU-First** (tối ưu hóa cho Intel Core i5-12450HX, 24 GB RAM, Windows 11, không yêu cầu GPU rời).

---

## 2. Bốn thí nghiệm bắt buộc (Experiment Matrix)

| Experiment ID | Kiến trúc | Phương pháp | Tầng Trainable | Tầng Bị đóng băng (Frozen) | Số tham số Trainable | % Trainable |
|:---|:---:|:---:|:---|:---|:---:|:---:|
| `resnet18_transfer` | ResNet18 | Transfer Learning | Duy nhất `model.fc` | `conv1`, `bn1`, `layer1`..`layer4` | 5,130 | 0.05% |
| `resnet18_finetune` | ResNet18 | Fine-Tuning | `layer4` + `fc` | `conv1`, `bn1`, `layer1`..`layer3` | 8,398,858 | 75.11% |
| `densenet121_transfer` | DenseNet121 | Transfer Learning | Duy nhất `classifier` | Toàn bộ `features` | 10,250 | 0.15% |
| `densenet121_finetune` | DenseNet121 | Fine-Tuning | `denseblock4` + `norm5` + `classifier` | `conv0`..`denseblock3` | 2,170,378 | 31.17% |

---

## 3. Cấu trúc thư mục dự án

```
c:\Project\lab 2/
├── README.md                          # Tài liệu hướng dẫn chính
├── requirements.txt                   # Danh sách thư viện phụ thuộc
├── .gitignore                         # Loại trừ dataset, checkpoint, cache
├── configs/
│   └── config.yaml                    # Cấu hình siêu tham số tập trung
├── data/
│   └── cifar-10-batches-py/           # Thư mục dữ liệu CIFAR-10
├── docs/
│   ├── project_description.md         # Mô tả bài toán và phạm vi
│   ├── project_design.md              # Thiết kế chi tiết các module & luồng dữ liệu
│   ├── experiment_plan.md             # Kế hoạch thực nghiệm và biến kiểm soát
│   ├── results.md                     # Báo cáo kết quả và phân tích
│   └── AI_Usage.md                    # Minh bạch sử dụng công cụ AI
├── src/
│   └── lab2/
│       ├── __init__.py
│       ├── config.py                  # Trình nạp và kiểm thực cấu hình
│       ├── data.py                    # Dataset, DataLoader, Transforms
│       ├── models.py                  # Xây dựng mô hình, freezing/unfreezing
│       ├── train.py                   # Vòng lặp train/val, checkpointing
│       ├── evaluate.py                # Đánh giá tập test, confusion matrix
│       ├── checkpoint.py              # Lưu/tải checkpoint an toàn
│       ├── tensorboard_logger.py      # Tích hợp SummaryWriter
│       └── utils.py                   # Thiết lập seed, device, vẽ biểu đồ
├── scripts/
│   ├── inspect_models.py              # Kiểm tra kiến trúc và đếm tham số
│   ├── run_experiments.py             # Chạy huấn luyện (từng bài hoặc tất cả)
│   └── evaluate_checkpoints.py        # Đánh giá lại các checkpoint đã lưu
├── tests/
│   ├── conftest.py
│   ├── test_data.py                   # Kiểm thử transforms và chia dữ liệu
│   ├── test_models.py                 # Kiểm thử cơ chế đóng băng/mở khóa
│   ├── test_training.py               # Kiểm thử 1 step gradient flow & config
│   └── test_checkpoint.py             # Kiểm thử lưu/tải checkpoint
├── checkpoints/                       # Nơi lưu các file checkpoint (*.pth)
├── runs/                              # Nơi lưu logs TensorBoard
└── outputs/
    ├── metrics/                       # Kết quả JSON sau khi test
    └── figures/                       # Hình ảnh ma trận nhầm lẫn (PNG)
```

---

## 4. Cài đặt môi trường

Dự án yêu cầu **Python 3.10+** (đã kiểm thử tương thích hoàn toàn trên Python 3.14).

Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

---

## 5. Dữ liệu và Pretrained Weights

- **CIFAR-10:** Hệ thống tự động phát hiện dữ liệu tại `data/cifar-10-batches-py/` hoặc tự động tải về khi chạy lần đầu.
- **Tập chia (Splitting):**
  - Train: 45.000 mẫu
  - Validation: 5.000 mẫu (chia từ train bằng `seed=42`)
  - Test: 10.000 mẫu (giữ nguyên không đổi)
- **Pretrained Weights:** Tự động tải từ kho chính thức của PyTorch `torchvision.models` (`ResNet18_Weights.DEFAULT`, `DenseNet121_Weights.DEFAULT`) và lưu cache vào thư mục người dùng (`~/.cache/torch/hub/checkpoints`).

---

## 6. Hướng dẫn thực thi

### 6.1. Chạy bộ kiểm thử tự động (Unit Tests)
Chạy toàn bộ 14 test cases trong vòng vài giây:
```bash
pytest -v
```

### 6.2. Kiểm tra kiến trúc và tham số mô hình
In bảng đối chiếu chi tiết số lượng tham số trainable/frozen của 4 thí nghiệm:
```bash
python scripts/inspect_models.py
```

### 6.3. Chạy kiểm thử nhanh Pipeline (Smoke Test)
Chạy kiểm tra toàn bộ luồng (train 1 epoch trên 120 mẫu, lưu checkpoint, đánh giá test và ghi log TensorBoard) chỉ trong vài giây:
```bash
python scripts/run_experiments.py --smoke-test --experiment resnet18_transfer
```

### 6.4. Chạy huấn luyện từng thí nghiệm
Chạy một thí nghiệm cụ thể theo cấu hình trong `configs/config.yaml`:
```bash
# 1. ResNet18 Transfer Learning
python scripts/run_experiments.py --experiment resnet18_transfer

# 2. ResNet18 Fine-Tuning
python scripts/run_experiments.py --experiment resnet18_finetune

# 3. DenseNet121 Transfer Learning
python scripts/run_experiments.py --experiment densenet121_transfer

# 4. DenseNet121 Fine-Tuning
python scripts/run_experiments.py --experiment densenet121_finetune
```

### 6.5. Chạy toàn bộ 4 thí nghiệm tuần tự
```bash
python scripts/run_experiments.py --all
```

### 6.6. Đánh giá Checkpoint trên tập Test
Đánh giá một checkpoint cụ thể:
```bash
python scripts/evaluate_checkpoints.py --checkpoint checkpoints/resnet18_transfer_best.pth
```
Hoặc đánh giá toàn bộ các checkpoint có hậu tố `*_best.pth` trong thư mục `checkpoints/`:
```bash
python scripts/evaluate_checkpoints.py --all
```

### 6.7. Khởi chạy và xem TensorBoard
Mở terminal và gõ:
```bash
tensorboard --logdir=runs
```
Sau đó mở trình duyệt và truy cập: `http://localhost:6006/`

---

## 7. Vị trí lưu trữ đầu ra

- **Checkpoints:** Lưu tại `checkpoints/` (`*_latest.pth` và `*_best.pth`).
- **TensorBoard Logs:** Lưu tại `runs/<experiment_id>/`.
- **Metrics JSON:** Lưu tại `outputs/metrics/<experiment_id>_test_metrics.json`.
- **Figures:** Lưu biểu đồ ma trận nhầm lẫn tại `outputs/figures/<experiment_id>_confusion_matrix.png`.

---

## 8. Các giới hạn đã biết (Known Limitations)

- Huấn luyện trên CPU với tập dữ liệu đầy đủ (50.000 ảnh) sẽ mất khoảng 30 - 60 phút mỗi thí nghiệm. Khuyến nghị sử dụng cờ `--smoke-test` hoặc giảm `batch_size` / `image_size` trong `config.yaml` nếu muốn kiểm thử nhanh tiến độ.
