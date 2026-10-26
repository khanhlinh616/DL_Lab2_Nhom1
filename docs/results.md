# Experimental Results & Analysis — Lab 2

## 1. Giới thiệu và Phương pháp Ghi nhận Kết quả
Báo cáo ghi nhận các kết quả đo đạc thực nghiệm trên môi trường tính toán CPU (Intel Core i5-12450HX, 24 GB RAM, Windows 11).
Các trường dữ liệu chưa hoàn thành đủ 5 epochs được ghi nhận bằng nhãn `*Chờ kết quả thực nghiệm*`.

---

## 2. Kết quả Khảo sát Kiến trúc và Phân tích Tham số (Architecture Inspection)

Được trích xuất trực tiếp thông qua lệnh `py scripts/inspect_models.py`:

| Experiment ID | Nhóm | Kiến trúc | Phương pháp | Tổng tham số (Total) | Tham số Trainable | Tham số Bị đóng băng (Frozen) | Tỷ lệ Trainable (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `resnet18_transfer` | `core` | ResNet18 | Transfer Learning | 11,181,642 | 5,130 | 11,176,512 | 0.05% |
| `resnet18_finetune` | `core` | ResNet18 | Fine-Tuning | 11,181,642 | 8,398,858 | 2,782,784 | 75.11% |
| `densenet121_transfer` | `core` | DenseNet121 | Transfer Learning | 6,964,106 | 10,250 | 6,953,856 | 0.15% |
| `densenet121_finetune` | `core` | DenseNet121 | Fine-Tuning | 6,964,106 | 2,170,378 | 4,793,728 | 31.17% |
| `resnet18_transfer_lr1e4` | `hparam` | ResNet18 | Transfer Learning | 11,181,642 | 5,130 | 11,176,512 | 0.05% |
| `resnet18_transfer_bs32` | `hparam` | ResNet18 | Transfer Learning | 11,181,642 | 5,130 | 11,176,512 | 0.05% |

### Nhận xét phân tích kiến trúc:
1. **ResNet18 Transfer Learning:** Chỉ duy nhất tầng kết nối đầy đủ cuối cùng (`fc.weight` và `fc.bias`) với $512 \times 10 + 10 = 5.130$ tham số được cập nhật. Toàn bộ các khối `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4` đều đóng băng hoàn toàn.
2. **ResNet18 Fine-Tuning:** Khi mở khóa khối tích chập sâu nhất `layer4` (gồm 2 BasicBlocks) cùng `fc`, số lượng tham số trainable tăng vọt lên 8.398.858 (chiếm 75.11% tổng mô hình), cho phép mô hình thích nghi các bộ lọc biểu diễn đặc trưng cấp cao theo dữ liệu CIFAR-10.
3. **DenseNet121 Transfer Learning:** Tầng phân loại `classifier` mới nhận đầu vào $1.024$ kênh, tạo ra $1.024 \times 10 + 10 = 10.250$ tham số cần huấn luyện (chiếm 0.15%).
4. **DenseNet121 Fine-Tuning:** Khác với ResNet, cấu trúc DenseNet có cơ chế nối ghép đặc trưng (feature concatenation). Việc mở khóa `denseblock4` (16 dense layers), `norm5` và `classifier` chỉ kích hoạt 2.170.378 tham số (chiếm 31.17%), giúp fine-tuning rất nhẹ nhàng và tiết kiệm tài nguyên tính toán hơn so với ResNet18.

---

## 3. Bảng Kết quả Thực nghiệm (Official Experiments Results)

Định dạng bảng tương thích hoàn toàn với script tự động `scripts/make_results_table.py`:

| Experiment ID | Group | Model | Method | Epochs | Batch Size | LR (Head/Backbone) | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Test Loss | Test Acc (%) | Runtime (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `resnet18_transfer` | core | resnet18 | transfer | 5 | 16 | 1e-3 / - | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* |
| `resnet18_finetune` | core | resnet18 | finetune | 5 | 16 | 1e-3 / 1e-4 | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* |
| `densenet121_transfer` | core | densenet121 | transfer | 5 | 16 | 1e-3 / - | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* |
| `densenet121_finetune` | core | densenet121 | finetune | 5 | 16 | 1e-3 / 1e-4 | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* |
| `resnet18_transfer_lr1e4` | hparam | resnet18 | transfer | 5 | 16 | 1e-4 / - | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* |
| `resnet18_transfer_bs32` | hparam | resnet18 | transfer | 5 | 32 | 1e-3 / - | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* |

> *Hướng dẫn cập nhật bảng:* Sau khi chạy full các thí nghiệm bằng lệnh `py scripts/run_experiments.py --group core` và `py scripts/run_experiments.py --group hparam`, chỉ cần thực thi lệnh:
> ```bash
> py scripts/make_results_table.py
> ```
> Script sẽ tự động đọc toàn bộ các file `outputs/metrics/*.json`, trích xuất số liệu chính xác và sinh ra bảng kết quả định dạng Markdown cùng file `outputs/metrics/summary.csv`.

---

## 4. Phân tích Chi tiết Kết quả Thực nghiệm

### 4.1. So sánh 4 Thí nghiệm Nòng cốt (Core Experiments)
- **Transfer Learning vs Fine-Tuning trên ResNet18:**
  - *Nhận xét kỳ vọng:* Fine-Tuning cho phép cập nhật tầng tích chập sâu `layer4` nên thường đạt Test Accuracy cao hơn từ 3% đến 6% so với Feature Extraction thuần túy, nhưng đòi hỏi thời gian huấn luyện mỗi epoch dài hơn do phải lan truyền ngược qua 8.4 triệu tham số.
  - *Số liệu thực tế:* *Chờ kết quả sau khi hoàn thành huấn luyện đủ epochs*
- **ResNet18 vs DenseNet121 trong Transfer Learning:**
  - *Nhận xét kỳ vọng:* DenseNet121 trích xuất đặc trưng có độ phong phú cao nhờ ghép nối các bản đồ đặc trưng từ tất cả các khối trước (dense connectivity), do đó phân loại trên vector đặc trưng 1024 chiều thường nhỉnh hơn một chút so với 512 chiều của ResNet18.
  - *Số liệu thực tế:* *Chờ kết quả sau khi hoàn thành huấn luyện đủ epochs*
- **Nguy cơ Overfitting:**
  - *Dấu hiệu:* Khoảng cách giữa Train Loss và Validation Loss. Trong Fine-Tuning với số tham số trainable lớn (75.11% ở ResNet18), nếu không có điều chuẩn đủ mạnh thì Train Loss sẽ giảm rất nhanh trong khi Val Loss có thể chững lại hoặc tăng nhẹ ở các epoch cuối.
  - *Số liệu thực tế:* *Chờ kết quả sau khi hoàn thành huấn luyện đủ epochs*

### 4.2. Khảo sát Siêu tham số (Hyperparameter Experiments)
- **Tốc độ học (Learning Rate: `1e-3` vs `1e-4`):**
  - Khi chỉ huấn luyện classification head (`fc`), tốc độ học `1e-3` thường hội tụ nhanh và đạt độ chính xác tốt hơn trong 5 epochs so với `1e-4` (vốn quá nhỏ khiến đầu phân loại chưa hội tụ hoàn toàn).
  - *Số liệu thực tế:* *Chờ kết quả sau khi hoàn thành huấn luyện đủ epochs*
- **Kích thước Batch (Batch Size: `16` vs `32`):**
  - Tăng batch size lên 32 giúp tăng thông lượng xử lý tính toán trên CPU (giảm số lần gọi forward/backward mỗi epoch), nhưng trên CPU có thể đòi hỏi bộ nhớ đệm lớn hơn và gradient có thể phẳng hơn.
  - *Số liệu thực tế:* *Chờ kết quả sau khi hoàn thành huấn luyện đủ epochs*

---

## 5. Phân tích Độ chính xác từng lớp & Ma trận Nhầm lẫn (Per-Class & Error Analysis)

### 5.1. Bảng Độ chính xác từng lớp (%)
Khi hoàn tất đánh giá tập Test qua `scripts/evaluate_checkpoints.py`, số liệu sẽ được trích xuất từ `outputs/metrics/*_test_metrics.json`:

| Lớp (Class) | `resnet18_transfer` | `resnet18_finetune` | `densenet121_transfer` | `densenet121_finetune` |
|---|:---:|:---:|:---:|:---:|
| 0. Airplane | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 1. Automobile | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 2. Bird | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 3. Cat | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 4. Deer | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 5. Dog | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 6. Frog | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 7. Horse | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 8. Ship | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |
| 9. Truck | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* | *Chờ chạy* |

### 5.2. Các cặp lớp dễ nhầm lẫn nhất (Top Confused Pairs)
Phân tích từ ma trận nhầm lẫn dự kiến:
1. **Cat (Mèo) $\leftrightarrow$ Dog (Chó):** Cả hai đều là động vật bốn chân có lông, hình thái tai, mắt và tư thế chụp trong ảnh 32x32 rất tương đồng khi bị nội suy phóng to.
2. **Automobile (Ô tô con) $\leftrightarrow$ Truck (Xe tải):** Đều là phương tiện giao thông trên đường, chứa các đặc trưng cạnh bánh xe và khung kim loại giống nhau.
3. **Airplane (Máy bay) $\leftrightarrow$ Bird (Chim):** Cả hai đều có nền trời xanh phía sau và hình dáng dang cánh ngang tương đồng.
- *Số liệu chi tiết:* Sinh tự động từ notebook [`notebooks/Lab2_Results_Analysis.ipynb`](../notebooks/Lab2_Results_Analysis.ipynb).

### 5.3. Hình ảnh Trực quan hóa
- Biểu đồ nhiệt Ma trận nhầm lẫn: Lưu tại thư mục [`outputs/figures/`](../outputs/figures/) (e.g. `resnet18_transfer_confusion_matrix.png`).
- Ảnh chụp màn hình TensorBoard (Loss và Accuracy theo Epoch): Lưu tại thư mục [`docs/figures/`](figures/).

---

## 6. Giới hạn Nghiên cứu Hiện tại (Current Limitations)

1. **Ràng buộc phần cứng CPU & Thời gian thực thi:** 
   Máy tính phát triển sử dụng CPU Intel Core i5-12450HX không có GPU rời NVIDIA. Thời gian huấn luyện 5 epochs trên 45.000 mẫu với ảnh 128x128 đòi hỏi khoảng 30 - 60 phút cho mỗi thí nghiệm.
2. **Nội suy phóng to ảnh (Bilinear Interpolation Artifacts):**
   Ảnh gốc CIFAR-10 có độ phân giải gốc $32 \times 32$. Khi phóng to 4 lần lên $128 \times 128$ để nạp vào các mô hình pretrained ImageNet, các chi tiết tần số cao bị mờ hóa (blurring), khiến mô hình khó phân biệt các đặc trưng vi mô của các lớp tương đồng.
3. **Giới hạn hạt giống ngẫu nhiên đơn lẻ (Single-Seed Limitation):**
   Mọi thực nghiệm hiện tại được kiểm soát ở một hạt giống cố định (`seed=42`). Do đó, số liệu đo lường chưa thể hiện khoảng tin cậy hay phương sai thống kê (mean ± standard deviation) qua nhiều lần chạy ngẫu nhiên độc lập.
4. **Hành vi cập nhật bộ đệm BatchNorm khi đóng băng Backbone (Thực nghiệm kiểm tra):**
   Đo lường thực nghiệm (`scripts/check_frozen_bn.py`) chỉ ra rằng trong chế độ `model.train()`, 100% bộ đệm BatchNorm (`running_mean`, `running_var`) của backbone đóng băng vẫn bị cập nhật theo batch CIFAR-10 hiện tại. Điều này có thể dẫn đến trôi phân phối đặc trưng đã học từ ImageNet.

---

## 7. Hướng phát triển và Cải tiến (Future Improvements)

1. **Đóng băng tường minh BatchNorm ở chế độ Eval (Frozen BatchNorm in Eval Mode):**
   Bổ sung phương thức chuyển riêng biệt các lớp `nn.BatchNorm2d` của backbone về `module.eval()` trong suốt vòng lặp train để giữ nguyên 100% thống kê tiền huấn luyện ImageNet.
2. **Đánh giá đa hạt giống (Multi-Seed Evaluation):**
   Thực thi mỗi thí nghiệm trên ít nhất 3 hạt giống khác nhau (`seed = [0, 42, 123]`) để tính giá trị trung bình và độ lệch chuẩn, khẳng định độ tin cậy thống kê của các kết luận so sánh.
3. **Bộ điều chỉnh tốc độ học (Learning Rate Scheduler):**
   Tích hợp `torch.optim.lr_scheduler.CosineAnnealingLR` hoặc `ReduceLROnPlateau` để giảm dần learning rate ở các epoch cuối, hỗ trợ mô hình hội tụ sâu hơn vào cực tiểu địa phương tối ưu.
4. **Mở rộng phần cứng:**
   Sử dụng môi trường GPU (như Google Colab T4 hoặc máy trạm CUDA) để tăng kích thước ảnh lên chuẩn ImageNet 224x224 và huấn luyện số lượng epochs lớn hơn (10 - 20 epochs).
