# Project Design — Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning

## 1. Kiến trúc hệ thống tổng thể (System Architecture)
Hệ thống được thiết kế theo mô hình **Modular PyTorch Architecture** (lấy cảm hứng từ *mrdbourke/pytorch-deep-learning* và các nguyên lý Software Quality Engineering):

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

---

## 2. Bảng giao diện các Module (Module Interfaces & Function Signatures)

Toàn bộ các hàm nòng cốt trong `src/lab2/` tuân thủ nghiêm ngặt type annotations và ràng buộc tham số:

| Module | Hàm / Lớp | Tham số đầu vào (Input) | Kết quả đầu ra (Output) | Ràng buộc & Ghi chú lỗi |
|---|---|---|---|---|
| `config.py` | `ExperimentConfig` (dataclass) | Các trường cấu hình thực nghiệm | Thực thể config đã xác thực | Ràng buộc dải giá trị: `lr > 0`, `val_split in (0, 0.5)`, `method in ['transfer', 'finetune']` |
| `config.py` | `load_config(path, experiment_name=None)` | `path: str`, `experiment_name: Optional[str]` | `ExperimentConfig` | Ném `FileNotFoundError` nếu sai đường dẫn, `KeyError` nếu thiếu trường |
| `config.py` | `list_experiments(path)` | `path: str` | `List[str]` | Trả về danh sách tên các cấu hình có trong YAML |
| `data.py` | `get_transforms(image_size, is_training)` | `image_size: int`, `is_training: bool` | `v2.Compose` | Chuẩn hóa ImageNet; chỉ thêm RandomCrop/Flip khi `is_training=True` |
| `data.py` | `get_cifar10_datasets(data_dir, val_split, seed, image_size)` | `data_dir: str`, `val_split: float`, `seed: int`, `image_size: int` | `Tuple[Dataset, Dataset, Dataset]` | Chia 45k train / 5k val bằng `torch.Generator().manual_seed(seed)` |
| `data.py` | `get_dataloaders(...)` | `train_ds, val_ds, test_ds, batch_size, num_workers, pin_memory` | `Tuple[DataLoader, DataLoader, DataLoader]` | `shuffle=True` cho train, `shuffle=False` cho val/test |
| `models.py` | `count_parameters(model)` | `model: nn.Module` | `Dict[str, int]` | Trả về `{'total': int, 'trainable': int, 'frozen': int}` |
| `models.py` | `build_model(model_name, method, num_classes, pretrained)` | `model_name: str`, `method: str`, `num_classes: int=10`, `pretrained: bool=True` | `nn.Module` | Ném `ValueError` nếu tên mô hình hoặc method không hỗ trợ |
| `train.py` | `train_one_epoch(...)` | `model, dataloader, criterion, optimizer, device` | `Tuple[float, float]` (loss, acc) | Gọi `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` |
| `train.py` | `validate(...)` | `model, dataloader, criterion, device` | `Tuple[float, float]` (loss, acc) | Chạy trong `torch.no_grad()`, `model.eval()` |
| `train.py` | `train(cfg, train_loader, val_loader, model=None, test_loader=None)` | `cfg: ExperimentConfig`, DataLoaders | `Dict[str, Any]` (history, best_acc, best_loss) | Hỗ trợ differential LR; log TensorBoard; checkpoint `_best.pth` |
| `checkpoint.py`| `save_checkpoint(state, is_best, checkpoint_dir, experiment_name)` | `state: dict, is_best: bool, ...` | `Tuple[Path, Optional[Path]]` | Tạo thư mục nếu chưa có; lưu `_latest.pth` và `_best.pth` |
| `checkpoint.py`| `load_checkpoint(checkpoint_path, model, optimizer=None)` | `checkpoint_path: str, model: nn.Module, ...` | `dict` (state) | Ném `FileNotFoundError` nếu checkpoint không tồn tại |
| `tensorboard_logger.py` | `TensorBoardLogger(...)` | `log_dir: str, experiment_name: str` | Đối tượng quản lý logger | Bọc `SummaryWriter` an toàn, hỗ trợ log scalars, graph, images, histograms |
| `evaluate.py` | `evaluate(model, dataloader, criterion, device, class_names=None)` | `model, dataloader, criterion, device, class_names` | `Dict[str, Any]` | Trả về `loss`, `accuracy`, `per_class_accuracy`, `confusion_matrix` |
| `evaluate.py` | `plot_confusion_matrix(cm, class_names, save_path)` | `cm: np.ndarray, class_names: list, save_path: str` | `None` | Vẽ seaborn heatmap với hiển thị số và tỷ lệ phần trăm |
| `utils.py` | `set_seed(seed=42)` | `seed: int` | `None` | Khóa seed cho random, numpy, torch, torch.cuda, benchmark |
| `utils.py` | `get_device()` | Không tham số | `torch.device` | Trả về `cuda` nếu có, fallback về `cpu` |
| `utils.py` | `save_metrics_json(metrics, filepath)` | `metrics: dict, filepath: str` | `None` | Lưu dict kết quả ra file JSON với UTF-8 indent=2 |
| `utils.py` | `save_history_json(history, filepath)` | `history: dict, filepath: str` | `None` | Lưu lịch sử train/val epoch ra file JSON |

---

## 3. Quyết định Thiết kế Kỹ thuật và Lý do (Design Decisions & Rationales)

### 3.1. Thuật toán tối ưu: Adam vs SGD with Momentum
- **Lựa chọn:** Adam (`lr=1e-3` cho transfer learning; `lr_backbone=1e-4`, `lr_head=1e-3` cho fine-tuning).
- **Lý do:**
  - Mô hình chạy trên môi trường CPU với số lượng epoch giới hạn (5 epoch).
  - Adam tính toán tốc độ học thích nghi cho từng trọng số dựa trên mô-men bậc một và bậc hai, giúp mô hình hội tụ nhanh chóng ngay từ những epoch đầu tiên mà không đòi hỏi phải tinh chỉnh kỳ công learning rate schedule hoặc momentum như SGD.
  - Phù hợp với khuyến nghị trong tài liệu thực hành Practice 2 (Slide 23–24).

### 3.2. Độ phân giải ảnh đầu vào: 128x128 thay vì 32x32 hoặc 224x224
- **Lựa chọn:** `image_size = 128`.
- **Lý do:**
  - Dữ liệu gốc CIFAR-10 có kích thước 32x32. Tuy nhiên, các kiến trúc pretrained trên ImageNet (ResNet18, DenseNet121) được thiết kế cho kích thước chuẩn 224x224 (có các tầng conv đầu kernel 7x7 stride 2 và max pooling 3x3 stride 2). Nếu đưa trực tiếp ảnh 32x32 vào, bản đồ đặc trưng sau các khối downsampling sẽ bị co cụm quá nhỏ (e.g. 1x1 hoặc nhỏ hơn) làm mất thông tin không gian.
  - Phóng to lên 224x224 trên CPU làm tăng khối lượng tính toán phép nhân-cộng (FLOPs) lên gấp khoảng 49 lần so với 32x32, khiến thời gian huấn luyện 1 epoch kéo dài hàng chục phút.
  - Kích thước 128x128 là điểm cân bằng kỹ thuật hoàn hảo giữa việc duy trì độ phân giải đặc trưng cho các kernel tích chập sâu và tốc độ xử lý trên CPU Intel Core i5.

### 3.3. Phân tách tập dữ liệu: 45.000 Train / 5.000 Validation độc lập
- **Lựa chọn:** Tách 10% từ tập train chính thức của CIFAR-10 với `seed=42`.
- **Lý do:**
  - Tập Test chính thức (10.000 ảnh) phải tuyệt đối được giữ nguyên làm dữ liệu kiểm thử không thiên vị (unbiased evaluation). Nếu sử dụng tập Test để theo dõi quá trình huấn luyện và chọn checkpoint thì checkpoint sẽ bị rò rỉ thông tin gián tiếp (data leakage / metric snooping).
  - Sử dụng `torch.Generator().manual_seed(seed)` đảm bảo việc phân tách hoàn toàn nhất quán và có thể tái lập trên mọi máy tính khác nhau.

### 3.4. Chiến lược Learning Rate vi phân (Differential Learning Rate) cho Fine-Tuning
- **Lựa chọn:** `lr_backbone = 1e-4`, `lr_head = 1e-3`.
- **Lý do:**
  - Backbone đã nắm giữ các bộ trích xuất đặc trưng bậc cao rất hữu ích từ ImageNet. Nếu cập nhật backbone với learning rate lớn ($10^{-3}$), các trọng số quý giá này sẽ bị "xóa sổ" (catastrophic forgetting).
  - Classification head mới được khởi tạo ngẫu nhiên nên cần tốc độ học lớn hơn ($10^{-3}$) để nhanh chóng thích nghi với không gian 10 nhãn mới của CIFAR-10.

### 3.5. Tiêu chí lưu Checkpoint tốt nhất
- **Lựa chọn:** Theo dõi `val_acc` cao nhất (với cơ chế dự phòng `val_loss`).
- **Lý do:**
  - Trong bài toán phân loại đa lớp cân bằng như CIFAR-10, `val_acc` phản ánh trực tiếp mục tiêu phân loại chính xác mẫu. Khi `val_acc` đạt mức cao nhất mới, trạng thái toàn diện của mô hình và optimizer được lưu tức thời thành `_best.pth`.

### 3.6. Phân nhóm Thực nghiệm: Core vs Hyperparameter
- **Lựa chọn:** Bổ sung trường `group: "core" | "hparam"` trong `configs/config.yaml`.
- **Lý do:**
  - Tách bạch 4 thực nghiệm cơ sở đối đầu (ResNet18 vs DenseNet121, Transfer vs Fine-tune) với 2 thực nghiệm khảo sát siêu tham số (Learning Rate và Batch Size).
  - CLI `scripts/run_experiments.py` hỗ trợ `--group core` hoặc `--group hparam` giúp linh hoạt thực thi mà không gây nhầm lẫn.

---

## 4. Rủi ro Kỹ thuật Đã biết và Phát hiện Thực nghiệm (Known Technical Risks & Empirical Findings)

### Hành vi cập nhật bộ đệm BatchNorm khi đóng băng Backbone (Thực nghiệm kiểm tra)
- **Bản chất kỹ thuật:**
  Trong PyTorch, khi thực hiện `param.requires_grad = False` cho một tầng tích chập hoặc tầng BatchNorm, ta chỉ đóng băng các tham số học được là trọng số (`weight`) và độ lệch (`bias`). Các tham số này sẽ có `grad is None` và không bị cập nhật bởi optimizer.
  Tuy nhiên, các tầng `BatchNorm2d` còn chứa các bộ đệm trạng thái (buffers) bao gồm `running_mean`, `running_var` và `num_batches_tracked`. Khi mô hình ở trạng thái `model.train()`, dù `requires_grad = False`, PyTorch vẫn tự động tính toán thống kê mini-batch hiện tại và cập nhật vào `running_mean` và `running_var` theo công thức exponential moving average.
- **Dữ liệu đo lường thực tế (`scripts/check_frozen_bn.py`):**
  - Trong chế độ `model.train()` (quy trình tiêu chuẩn của bài Lab):
    - **ResNet18:** 60/60 bộ đệm BatchNorm trong backbone (100%) bị thay đổi giá trị sau 1 batch forward pass.
    - **DenseNet121:** 363/363 bộ đệm BatchNorm trong backbone (100%) bị thay đổi giá trị sau 1 batch forward pass.
  - Trong chế độ `model.eval()`:
    - 0/60 bộ đệm của ResNet18 và 0/363 bộ đệm của DenseNet121 thay đổi (0%).
- **Tác động & Giải pháp tương lai:**
  - Việc cập nhật thống kê running mean/var của CIFAR-10 lên các tầng đóng băng vốn được huấn luyện trên ImageNet có thể gây ra hiện tượng trôi phân phối đặc trưng (covariate shift) ở các tầng đầu.
  - Dự án hiện giữ nguyên quy trình PyTorch chuẩn mực của bài thực hành (`model.train()`) để đảm bảo tính nhất quán với giáo trình, đồng thời ghi nhận và chứng minh rõ ràng hiện tượng này trong tài liệu kỹ thuật và bộ test tự động (`tests/test_frozen_bn.py`).
  - Hướng cải tiến: Cung cấp tùy chọn gọi tường minh `module.eval()` riêng cho các tầng BatchNorm bị đóng băng trong giai đoạn train.
