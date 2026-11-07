# ĐẶC TẢ & BÁO CÁO KỸ THUẬT: LAB 2 — PRETRAINED MODELS, TRANSFER LEARNING & FINE-TUNING

```
===================================================================================================
TÓM TẮT DỰ ÁN (EXECUTIVE SUMMARY)
Mục tiêu    : Chuyển giao tri thức ImageNet-1K sang phân loại 10 lớp CIFAR-10.
Mô hình     : ResNet-18 (Residual Addition) vs. DenseNet-121 (Dense Concatenation).
Phương pháp : Transfer Learning (Feature Extraction) vs. Fine-Tuning (Differential LR).
Môi trường  : CPU-First / Windows 11 / Python 3.14 / PyTorch 2.14.1+cpu.
Kết quả     : Fine-Tuning đạt Test Acc 90.24% (ResNet-18) và 90.65% (DenseNet-121). 53/53 tests PASS.
===================================================================================================
```

---

## 1. TỔNG QUAN & MỤC TIÊU BÀI TOÁN

- **Bài toán:** Phân loại ảnh màu 10 lớp trên tập dữ liệu chuẩn **CIFAR-10**.
- **Tập nguồn (Source Dataset):** **ImageNet-1K** (1,28 triệu ảnh, 1.000 danh mục) $\to$ trích xuất đặc trưng phân cấp sẵn có.
- **Tập đích (Target Dataset):** **CIFAR-10** (60.000 ảnh $32 \times 32 \times 3$, 10 lớp cân bằng: 6 loài vật + 4 phương tiện).
- **Thách thức:** Huấn luyện Deep CNN từ đầu (from scratch) trên ảnh nhỏ dễ bị **Overfitting** và tốn tài nguyên.
- **Giải pháp:** Áp dụng **Transfer Learning** (đóng băng backbone) và **Fine-Tuning** (tinh chỉnh tầng sâu với Differential LR).

---

## 2. VÍ DỤ ĐẦU VÀO - ĐẦU RA CỤ THỂ (EXAMPLE IN - OUT)

Hệ thống nhận ảnh thô, tiền xử lý và trả về nhãn phân loại độc lập ngoài terminal/file:

```
[INPUT: ẢNH THÔ CIFAR-10]
- Kích thước: 32 x 32 x 3 (uint8, [0, 255])
- Nhãn thực tế (Ground Truth): "dog" (Lớp 5)
        │
        ▼ [PIPELINE TIỀN XỬ LÝ]
1. Resize: Song tuyến (Bilinear) lên 128 x 128 pixels (khớp Receptive Field).
2. Data Augmentation (chỉ Train): RandomCrop(128, pad=16), RandomHorizontalFlip(p=0.5).
3. ToTensor(): Chuyển kiểu sang float32 [0.0, 1.0], đổi trục thành [3, 128, 128].
4. Normalize: Chuẩn hóa ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
        │
        ▼ [TENSOR ĐƯA VÀO MODEL]
- Shape: [1, 3, 128, 128] (Kiểu: torch.float32)
        │
        ▼ [FORWARD PASS QUA MÔ HÌNH (DENSENET-121 FINE-TUNE)]
- Logits thô: [-1.42, -2.10, 0.15, 2.38, 0.81, 6.94, -0.22, 1.10, -3.05, -1.88]
- Softmax Probabilities:
  airplane: 0.02%  | automobile: 0.01% | bird: 0.11%   | cat: 1.05%
  deer    : 0.22%  | dog       : 97.42%| frog: 0.08%   | horse: 0.29%
  ship    : 0.01%  | truck     : 0.01%
        │
        ▼ [OUTPUT DỰ ĐOÁN CUỐI CÙNG]
- Predicted Label : "dog" (Lớp 5)
- Confidence Score: 97.42%
- Kết quả đối soát : CHÍNH XÁC (Trùng khớp 100% với Ground Truth)
```

---

## 3. LÝ DO LỰA CHỌN MÔ HÌNH CHO BÀI TOÁN CIFAR-10

1. **Tính đại diện kiến trúc:**
   - *ResNet-18:* Đại diện cho **đường tắt cộng dư (Residual Skip Connections)** $y = \mathcal{F}(x) + x$, triệt tiêu suy biến gradient.
   - *DenseNet-121:* Đại diện cho **ghép nối kênh đặc trưng (Feature Concatenation)** $x_\ell = H([x_0, \dots, x_{\ell-1}])$, tối đa hóa tái sử dụng đặc trưng.
2. **Tương thích quy mô CIFAR-10 & Chống Overfitting:**
   - CIFAR-10 có kích thước nhỏ ($32 \times 32$), 45.000 ảnh train.
   - Các mô hình lớn (ResNet-50/101: 25 - 45M tham số; DenseNet-201: 20M tham số) quá nặng, gây học vẹt. ResNet-18 (11,2M) và DenseNet-121 (7,0M) là lựa chọn tối ưu, đủ sâu để học ngữ nghĩa nhưng kiểm soát tốt phương sai.
3. **Đánh đổi Tham số vs. Băng thông tính toán:**
   - DenseNet-121 tiết kiệm **37,7% tham số** so với ResNet-18 (6,96M vs 11,18M).
   - Tuy nhiên, thời gian huấn luyện DenseNet-121 lại lâu hơn (10m42s vs 7m15s) do thao tác ghép nối tensor chiếm nhiều băng thông RAM.
4. **Cấu trúc phân tầng lý tưởng:** Cả 2 mô hình gồm 4 giai đoạn rõ ràng, dễ dàng đóng băng toàn bộ hoặc mở khóa chọn lọc 1 khối sâu nhất khi fine-tune.

---

## 4. BƯỚC 3: KHÁM PHÁ KIẾN TRÚC MÔ HÌNH (EXPLORE THE MODEL ARCHITECTURE)

Đã hoàn thành toàn bộ Bước 3 và cung cấp công cụ dòng lệnh [`scripts/inspect_models.py`](../scripts/inspect_models.py) để chạy trực tiếp:

### 4.1. Lệnh thực thi ngoài dòng lệnh (CLI Commands)
```bash
# In toàn bộ cấu trúc phân tầng PyTorch ra màn hình (print(model))
py scripts/inspect_models.py --print-model all

# Phân tích chi tiết từng khối tầng (named_children) và trạng thái Frozen/Trainable
py scripts/inspect_models.py --layers all --method finetune

# Kiểm tra chi tiết từng tensor tham số (named_parameters, shape, numel, requires_grad)
py scripts/inspect_models.py --params resnet18 --method transfer
```

### 4.2. Bảng phân rã cấu trúc tầng (Layer Breakdown)
| Khối / Tầng | ResNet-18 (Kiểu / Số tham số) | DenseNet-121 (Kiểu / Số tham số) | Trạng thái khi Fine-Tuning |
|:---|:---|:---|:---:|
| **Stem** | `conv1` + `bn1` (9.536 params) | `conv0` + `norm0` (9.536 params) | Đóng băng (`requires_grad=False`) |
| **Stage 1** | `layer1` (147.968 params) | `denseblock1` + `transition1` (368.320 params) | Đóng băng (`requires_grad=False`) |
| **Stage 2** | `layer2` (525.568 params) | `denseblock2` + `transition2` (1.051.776 params) | Đóng băng (`requires_grad=False`) |
| **Stage 3** | `layer3` (2.099.712 params) | `denseblock3` + `transition3` (3.364.096 params) | Đóng băng (`requires_grad=False`) |
| **Stage 4** | `layer4` (8.393.728 params) | `denseblock4` + `norm5` (2.160.128 params) | **Mở khóa (`requires_grad=True`)** |
| **Head** | `fc`: `Linear(512, 10)` (5.130 params) | `classifier`: `Linear(1024, 10)` (10.250 params) | **Mở khóa (`requires_grad=True`)** |

### 4.3. Bảng phân bổ tham số (Parameter Profiling)
| Cấu hình Thí nghiệm | Tổng tham số | Tham số Huấn luyện (Trainable) | Tham số Đóng băng (Frozen) | Tỷ lệ Trainable |
|:---|:---:|:---:|:---:|:---:|
| `resnet18_transfer` | 11.181.642 | **5.130** | 11.176.512 | **0,05%** |
| `resnet18_finetune` | 11.181.642 | **8.398.858** | 2.782.784 | **75,11%** |
| `densenet121_transfer` | 6.964.106 | **10.250** | 6.953.856 | **0,15%** |
| `densenet121_finetune` | 6.964.106 | **2.170.378** | 4.793.728 | **31,17%** |

---

## 5. HAI CHIẾN LƯỢC THÍCH NGHI MÔ HÌNH

- **Chiến lược 1 — Transfer Learning (Feature Extraction):**
  - Đóng băng toàn bộ backbone (`requires_grad = False`).
  - Chỉ huấn luyện Classifier Head mới (`Linear(D, 10)`) với $\eta = 10^{-3}$.
  - Ưu điểm: Huấn luyện nhanh (5 - 9 phút), tốn ít bộ nhớ.
- **Chiến lược 2 — Fine-Tuning (Tinh chỉnh tầng sâu):**
  - Đóng băng các tầng nông; mở khóa tầng sâu nhất (`layer4` / `denseblock4`) và Head.
  - Áp dụng **Differential Learning Rate**: $\eta_{backbone} = 10^{-4} = \frac{1}{10} \eta_{head} = 10^{-3}$.
  - Mục đích: Chống **quên thảm họa (Catastrophic Forgetting)** đối với tri thức ImageNet.

---

## 6. KẾT QUẢ THỰC NGHIỆM ĐẦY ĐỦ (XUẤT RA NGOÀI TỪ `summary.csv`)

Số liệu đo đạc thực tế trích xuất trực tiếp từ [`outputs/metrics/summary.csv`](../outputs/metrics/summary.csv):

| STT | Thí nghiệm ID | Mô hình | Phương pháp | LR (Head / Backbone) | Batch | Trainable Params | Best Val Acc (epoch) | Test Loss | **Test Acc (%)** | Tổng thời gian |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `resnet18_transfer` *(Base)* | ResNet-18 | Transfer | 1e-3 / — | 16 | 5.130 (0,05%) | 77.94% (ep 2) | 0.6644 | **77.98%** | 05m 35s |
| 2 | `resnet18_finetune` | ResNet-18 | Fine-Tuning | 1e-3 / 1e-4 | 16 | 8.398.858 (75,11%) | 91.68% (ep 4) | 0.2989 | **90.24%** | 07m 15s |
| 3 | `densenet121_transfer` | DenseNet-121 | Transfer | 1e-3 / — | 16 | 10.250 (0,15%) | 81.84% (ep 5) | 0.5560 | **81.34%** | 09m 03s |
| 4 | `densenet121_finetune` | DenseNet-121 | Fine-Tuning | 1e-3 / 1e-4 | 16 | 2.170.378 (31,17%) | 90.78% (ep 5) | 0.2982 | **90.65%** | 10m 42s |
| 5 | `resnet18_transfer_lr1e4` | ResNet-18 | Transfer | 1e-4 / — | 16 | 5.130 (0,05%) | 77.56% (ep 5) | 0.6643 | **78.32%** | 06m 03s |
| 6 | `resnet18_transfer_bs32` | ResNet-18 | Transfer | 1e-3 / — | 32 | 5.130 (0,05%) | 78.92% (ep 4) | 0.6276 | **79.04%** | 05m 21s |

### Kết luận đối chứng quan trọng:
1. **Fine-Tuning vượt trội Transfer Learning:** Tăng **+12,26%** trên ResNet-18 và **+9,31%** trên DenseNet-121, đưa độ chính xác vượt mốc 90%.
2. **DenseNet-121 trích xuất tốt hơn:** Ở Transfer Learning, DenseNet-121 (81,34%) vượt ResNet-18 (77,98%) **+3,36%** nhờ ghép kênh đa tầng.
3. **Khảo sát siêu tham số:** Giảm LR xuống $10^{-4}$ tăng nhẹ Test Acc (+0,34%); tăng Batch size lên 32 rút ngắn thời gian chạy (~14s) và tăng nhẹ Test Acc (+1,06%).

---

## 7. HƯỚNG DẪN THỰC THI QUA GIAO DIỆN DÒNG LỆNH (CLI)

Toàn bộ quy trình có thể chạy độc lập ngoài dòng lệnh không cần mở notebook:

```bash
# 1. Chạy toàn bộ 53 bài kiểm thử tự động (Unit Tests)
py -m pytest -q

# 2. Khám phá kiến trúc mô hình (Bước 3)
py scripts/inspect_models.py --print-model all
py scripts/inspect_models.py --layers all

# 3. Chạy thử nghiệm khói nhanh (Smoke Test - 1 epoch, subset 120 mẫu)
py scripts/run_experiments.py --smoke-test --experiment resnet18_transfer

# 4. Kiểm chứng hành vi BatchNorm buffer khi đóng băng backbone
py scripts/check_frozen_bn.py

# 5. Đánh giá tất cả checkpoint trên tập Test độc lập
py scripts/evaluate_checkpoints.py --all

# 6. Tái tổng hợp bảng kết quả định lượng
py scripts/make_results_table.py
```

---

## 8. GIỚI HẠN & HƯỚNG CẢI TIẾN (LIMITATIONS & ROADMAP)

- **Giới hạn hiện tại:**
  1. *Hạt giống đơn lẻ:* Mới chạy trên `seed = 42`, chưa đo được phương sai ($\mu \pm \sigma$).
  2. *Số epochs:* Huấn luyện 5 epochs; `densenet121_transfer` đạt best val ở epoch 5 nên có thể chưa hội tụ hoàn toàn.
  3. *Độ phân giải:* Resize ảnh lên $128 \times 128$ thay vì chuẩn $224 \times 224$ của ImageNet để tiết kiệm $67,3\%$ FLOPs trên CPU.
- **Định hướng cải tiến:**
  1. Đánh giá đa hạt giống (`seed = [42, 123, 999]`) để khẳng định độ tin cậy thống kê.
  2. Tích hợp `CosineAnnealingLR` để điều chỉnh tốc độ học mượt mà hơn.
  3. Mở rộng huấn luyện 15 - 20 epochs trên GPU với độ phân giải $224 \times 224$.

---

## 9. ĐỐI CHIẾU TIÊU CHUẨN HOÀN THÀNH (DEFINITION OF DONE - DoD)

| Tiêu chuẩn hoàn thành | Trạng thái | Minh chứng kỹ thuật |
|:---|:---:|:---|
| **Đủ yêu cầu (Completeness)** | **ĐẠT** | 2 mô hình (ResNet-18, DenseNet-121), 2 phương pháp (Transfer, Fine-tune), 2 siêu tham số (LR, Batch). |
| **Code chạy (Execution)** | **ĐẠT** | Toàn bộ script trong `scripts/` và module `src/lab2/` chạy Exit Code 0, không runtime crash. |
| **Test passed (100% Tests)** | **ĐẠT** | 53/53 bài kiểm thử `pytest` vượt qua trong ~10 giây (`53 passed in 10.84s`). |
| **Bảo mật (No Secrets)** | **ĐẠT** | Không lộ API key, token, mật khẩu hay biến môi trường nhạy cảm. |
| **Đánh giá khách quan** | **ĐẠT** | Tập Test 10.000 ảnh cách ly 100%, chỉ nạp 1 lần ở cuối trên Best Checkpoint. |
| **Lưu trữ an toàn** | **ĐẠT** | Áp dụng cơ chế lưu nguyên tử `atomic_save` qua tệp tạm `.tmp` chống hỏng dữ liệu. |
| **Đồng bộ kho lưu trữ** | **ĐẠT** | Nhánh `main` đồng bộ hoàn toàn với GitHub `vongocgiavy/DL_Lab-2`. |
| **Tài liệu đầy đủ** | **ĐẠT** | Hệ thống tài liệu trong `/docs` đầy đủ bảng số liệu, lệnh CLI và minh chứng rõ ràng. |
