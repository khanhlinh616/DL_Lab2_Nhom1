# Project Design — Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning

## 1. Kiến trúc hệ thống tổng thể (System Architecture)
Hệ thống được thiết kế theo mô hình **Modular PyTorch Architecture** (lấy cảm hứng từ *mrdbourke/pytorch-deep-learning* và các nguyên lý của Software Quality Engineering):

```
                        ┌────────────────────────┐
                        │   configs/config.yaml  │
                        └───────────┬────────────┘
                                    │
                                    ▼
                             [src/lab2/config.py]
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │                              │                              │
     ▼                              ▼                              ▼
[src/lab2/data.py]         [src/lab2/models.py]        [src/lab2/tensorboard_logger.py]
(CIFAR-10, Split,          (ResNet18 / DenseNet121,    (SummaryWriter, Scalars,
 Augment/Normalize)         Freezing / Unfreezing)      Graph, Histograms)
     │                              │                              │
     └──────────────────────────────┼──────────────────────────────┘
                                    │
                                    ▼
                          [src/lab2/train.py]
                        (Train & Validation Loops)
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
      [src/lab2/checkpoint.py]              [src/lab2/evaluate.py]
    (Save Best / Latest Model)             (Unbiased Test Evaluation,
                                            Confusion Matrix, Metrics JSON)
```

## 2. Chi tiết các thành phần (Module Details)

### 2.1. `src/lab2/config.py` (Cấu hình và Kiểm thực)
- Sử dụng Python `dataclass` (`ExperimentConfig`) ánh xạ trực tiếp từ file `configs/config.yaml`.
- Cơ chế xác thực chặt chẽ (`validate()`):
  - Kiểm tra `model_name` trong `['resnet18', 'densenet121']`.
  - Kiểm tra `method` trong `['transfer', 'finetune']`.
  - Kiểm tra số nguyên dương cho `epochs`, `batch_size`, `num_classes`, `image_size`.
  - Kiểm tra dải hợp lệ cho `learning_rate > 0`, `val_split` nằm trong khoảng $(0, 0.5)$.

### 2.2. `src/lab2/data.py` (Quản lý Dữ liệu và Pipeline Tiền xử lý)
- **Tải và Cache:** Tự động kiểm tra tính sẵn sàng của CIFAR-10 trong `./data`. Xử lý chứng chỉ SSL ngoại lệ trên Windows đảm bảo tải tin cậy.
- **Phân tách Tập dữ liệu (Reproducible Data Splitting):**
  - Tập huấn luyện gốc 50.000 ảnh được chia thành 45.000 ảnh (Train) và 5.000 ảnh (Validation) với tỷ lệ `val_split=0.1`.
  - Sử dụng `torch.Generator().manual_seed(seed)` để phân chia ngẫu nhiên nhưng có thể tái lập tuyệt đối trên mọi lần chạy.
  - Tập kiểm thử chính thức (10.000 ảnh) được giữ độc lập, không tham gia vào bất kỳ khâu huấn luyện hay chọn checkpoint nào.
- **Biến đổi Dữ liệu (Transforms):**
  - Chuẩn hóa ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` phù hợp với các trọng số pretrained.
  - Tập Train: `Resize((image_size, image_size))` $\to$ `RandomCrop` $\to$ `RandomHorizontalFlip(p=0.5)` $\to$ `ToTensor()` $\to$ `Normalize()`.
  - Tập Validation & Test: `Resize((image_size, image_size))` $\to$ `ToTensor()` $\to$ `Normalize()`. Tuyệt đối không áp dụng random augmentation trên validation/test.

### 2.3. `src/lab2/models.py` (Mô hình và Cơ chế Đóng băng/Mở khóa)
Mô hình khởi tạo từ `torchvision.models` với `weights=ResNet18_Weights.DEFAULT` và `DenseNet121_Weights.DEFAULT`.

| Mô hình | Phương pháp | Cơ chế Đóng băng / Mở khóa | Số tham số Trainable | % Trainable |
|:---|:---|:---|:---:|:---:|
| **ResNet18** | Transfer Learning | Đóng băng toàn bộ backbone (`conv1`..`layer4`), chỉ mở khóa `fc` | 5,130 | 0.05% |
| **ResNet18** | Fine-Tuning | Đóng băng `conv1`..`layer3`, mở khóa `layer4` và `fc` | 8,398,858 | 75.11% |
| **DenseNet121** | Transfer Learning | Đóng băng toàn bộ `features`, chỉ mở khóa `classifier` | 10,250 | 0.15% |
| **DenseNet121** | Fine-Tuning | Đóng băng `conv0`..`denseblock3`, mở khóa `denseblock4`, `norm5`, `classifier` | 2,170,378 | 31.17% |

### 2.4. `src/lab2/train.py` (Vòng lặp Huấn luyện & Đánh giá)
- **Quy trình Huấn luyện (Slide 8-9):**
  1. `data, labels = data.to(device), labels.to(device)`
  2. `optimizer.zero_grad()`
  3. `outputs = model(data)`
  4. `loss = criterion(outputs, labels)`
  5. `loss.backward()`
  6. `optimizer.step()`
- **Quy trình Validation (Slide 10-11):**
  - Bật `model.eval()`, bọc trong `with torch.no_grad():`, tính toán `running_loss` và `correct_predictions`, sau đó trả về `model.train()`.
- **Differential Learning Rates:**
  - Trong kịch bản Fine-Tuning, classifier mới dùng learning rate cao hơn (e.g. $10^{-3}$) còn các tầng tích chập sâu của backbone dùng learning rate nhỏ hơn (e.g. $10^{-4}$) để không phá vỡ tri thức đã học trước đó.

### 2.5. `src/lab2/checkpoint.py` (Lưu trữ và Khôi phục Mô hình)
- Lưu trữ toàn diện thông tin checkpoint:
  - `model_state_dict`: Trọng số của mô hình.
  - `optimizer_state_dict`: Trạng thái optimizer (momentum, learning rate).
  - `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`.
  - `best_metric`, `best_epoch`, `config` snapshot.
- Lựa chọn checkpoint tốt nhất: So sánh dựa trên validation accuracy cao nhất (hoặc validation loss thấp nhất) và lưu ra file `*_best.pth`.

### 2.6. `src/lab2/tensorboard_logger.py` (Tích hợp TensorBoard)
- Theo dõi thời gian thực các đại lượng:
  - Scalars: `Loss/train`, `Loss/val`, `Accuracy/train`, `Accuracy/val`.
  - Graphs: `writer.add_graph(model, dummy_input)`.
  - Sample images: Trực quan hóa batch ảnh đầu tiên bằng grid.
  - Histograms: Phân phối trọng số của các lớp trainable qua từng epoch.
- Đóng writer an toàn với `writer.close()`.

### 2.7. `src/lab2/evaluate.py` (Đánh giá Độc lập trên Tập Test)
- Nạp checkpoint tốt nhất đã lưu.
- Tính toán tổng thể `test_loss`, `test_accuracy`.
- Tính độ chính xác theo từng lớp (`per_class_accuracy`).
- Sinh ma trận nhầm lẫn (`confusion_matrix`) và xuất biểu đồ nhiệt (heatmap PNG) vào `outputs/figures/`.
- Xuất file kết quả JSON vào `outputs/metrics/`.

## 3. Thiết kế Kiểm thử (Test Design)
- Áp dụng phương pháp Test-Driven Development (TDD).
- Bộ unit test sử dụng dữ liệu giả lập (synthetic data), chạy độc lập và không phụ thuộc vào việc tải dataset lớn:
  1. `test_data.py`: Kiểm tra shape tensor sau transform, kiểm tra tính không giao thoa giữa train và val split, kiểm tra nhãn CIFAR-10.
  2. `test_models.py`: Kiểm tra số lượng tham số trainable/frozen của 4 cấu hình, kiểm tra forward pass trả về tensor `(B, 10)`.
  3. `test_training.py`: Kiểm tra 1 step gradient flow: khẳng định các lớp bị đóng băng không sinh gradient (`grad is None`) và trọng số không đổi, trong khi các lớp trainable được cập nhật thành công.
  4. `test_checkpoint.py`: Kiểm tra tính toàn vẹn của checkpoint sau khi lưu và nạp lại (`torch.equal`).
