# MODEL DOCUMENTATION (MODEL_DOCUMENTATION.MD)

Tài liệu kỹ thuật đặc tả chi tiết kiến trúc mô hình, nguồn gốc tham số, cơ chế chuyển giao tri thức, luồng dữ liệu vào/ra và quy trình kiểm chứng thực nghiệm trong khuôn khổ Laboratory 2.

---

## 1. CREATED DATE
- **2026-10-27**

---

## 2. UPDATED DATE
- **2026-11-09**

---

## 3. PURPOSE

- **Objective:** Nghiên cứu, thực nghiệm và so sánh đối chứng có hệ thống khả năng chuyển giao tri thức (Transfer Learning) từ các mô hình học sâu thị giác tiền huấn luyện trên ImageNet-1K sang bài toán phân loại ảnh CIFAR-10 trong môi trường CPU-First, giải quyết triệt để bài toán chống quá khớp (Overfitting) và tối ưu hóa tài nguyên tính toán.
- **Task:** Xây dựng, điều chỉnh hai kiến trúc ResNet-18 và DenseNet-121; so sánh 2 phương pháp thích nghi (Transfer Learning với Feature Extraction vs. Fine-Tuning với Differential Learning Rates); khảo sát độ nhạy siêu tham số (Learning Rate $10^{-3}$ vs. $10^{-4}$, Batch Size $16$ vs. $32$); thực thi hoàn chỉnh **Bước 3: Khám phá kiến trúc mô hình (Explore the Model Architecture)** qua việc in cấu trúc phân tầng, xác định layers và tham số.
- **Expected Outcome:** Đạt độ chính xác kiểm thử Test Accuracy $> 90\%$ trên tập Test 10.000 ảnh độc lập khi Fine-Tuning; làm sáng tỏ sự đánh đổi giữa hiệu quả tham số và chi phí tính toán thực tế; đảm bảo Zero Data Leakage; lưu trữ checkpoint an toàn nguyên tử (`atomic_save`); vượt qua 100% bài kiểm thử tự động `pytest`.

---

## 4. INITIAL MODEL & SOURCE INTEGRITY

- **Model Names & Versions:**
  - **ResNet-18:** Mạng nơ-ron dư 18 tầng (Residual Networks - 18 layers).
  - **DenseNet-121:** Mạng nơ-ron tích chập kết nối dày đặc 121 tầng (Densely Connected Convolutional Networks - 121 layers).
- **Original Research Papers & Authors:**
  - *ResNet:* Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *"Deep Residual Learning for Image Recognition"*, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.
  - *DenseNet:* Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. *"Densely Connected Convolutional Networks"*, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 4700-4708.
- **Pretrained Weights & Accuracy:**
  - `ResNet18_Weights.IMAGENET1K_V1` (Top-1 Accuracy: 69,758%, Top-5 Accuracy: 89,078% trên ImageNet-1K).
  - `DenseNet121_Weights.IMAGENET1K_V1` (Top-1 Accuracy: 74,434%, Top-5 Accuracy: 91,972% trên ImageNet-1K).
- **Official Repository & Framework:** Thư viện học sâu chuẩn `torchvision.models` (Torchvision 0.20.1+cpu / PyTorch 2.14.1+cpu).
- **Model License:** BSD 3-Clause License (PyTorch / Torchvision Open Source Ecosystem).
- **Initialization Strategy:**
  - *Backbone:* Nạp nguyên vẹn trọng số tiền huấn luyện ImageNet-1K V1.
  - *Classifier Head mới (10 lớp):* Khởi tạo ngẫu nhiên theo phân phối **Kaiming Uniform** (`torch.nn.init.kaiming_uniform_`):
    $$\text{weight} \sim \mathcal{U}(-\sqrt{k}, \sqrt{k}), \quad \text{bias} \sim \mathcal{U}(-\sqrt{k}, \sqrt{k}), \quad \text{với } k = \frac{1}{\text{in\_features}}$$
  - *Chiến lược Freeze / Unfreeze:*
    - **Transfer Learning (Feature Extraction):** Đóng băng toàn bộ các tham số của backbone (`requires_grad = False`); chỉ mở khóa huấn luyện duy nhất lớp phân loại mới Classifier Head (`requires_grad = True`).
    - **Fine-Tuning:** Đóng băng các tầng nông (Stem, Stages 1–3); mở khóa toàn bộ khối đặc trưng sâu nhất (`layer4` trên ResNet-18 hoặc `denseblock4` + `norm5` trên DenseNet-121) cùng Classifier Head (`requires_grad = True`), kết hợp cơ chế tốc độ học phân tầng Differential Learning Rates:
      $$\eta_{\text{backbone}} = 0.1 \times \eta_{\text{head}} = 0.0001, \quad \eta_{\text{head}} = 0.001$$

---

## 5. MODEL ARCHITECTURE

### 5.1. Tổng quan kiến trúc (Architecture Overview)
- **ResNet-18:** Đại diện cho triết lý **đường tắt cộng dồn (Residual Identity Shortcuts)**:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
  Cho phép gradient chảy trực tiếp qua các tầng mà không bị suy biến, hỗ trợ tối ưu mạng sâu hiệu quả.
- **DenseNet-121:** Đại diện cho triết lý **kết nối dày đặc (Dense Feature Concatenation)**:
  $$\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}])$$
  Mỗi tầng nhận toàn bộ bản đồ đặc trưng của các tầng trước làm đầu vào qua phép ghép nối tensor (concatenation), tối đa hóa khả năng tái sử dụng đặc trưng với hệ số tăng trưởng (Growth Rate) $k = 32$.
- **Lý do lựa chọn cho bài toán CIFAR-10:** CIFAR-10 có kích thước nhỏ ($32 \times 32$), 45.000 ảnh huấn luyện. Các kiến trúc lớn (ResNet-50/101: 25.5M - 44.5M tham số; DenseNet-201: 20M tham số) quá cồng kềnh, dễ dẫn đến hiện tượng học vẹt (Overfitting). ResNet-18 (11.18M tham số) và DenseNet-121 (6.96M tham số) là hai kiến trúc tối ưu, đủ sâu để học ngữ nghĩa phức tạp nhưng kiểm soát tốt phương sai mô hình.
- **Quy trình Bước 3 (Explore the Model Architecture):** Được thực thi hoàn chỉnh và kiểm chứng độc lập qua công cụ CLI [`scripts/inspect_models.py`](../scripts/inspect_models.py) (`--print-model`, `--layers`, `--params`).

### 5.2. Cấu trúc Backbone chi tiết

#### ResNet-18 Backbone:
1. **Stem:** `conv1` ($7 \times 7$, stride 2, padding 3, 64 filters) + `bn1` + `relu` + `maxpool` ($3 \times 3$, stride 2, padding 1) $\to$ **9.536 tham số**.
2. **Stage 1 (`layer1`):** 2 khối `BasicBlock` liên tiếp (64 kênh, không giảm mẫu) $\to$ **147.968 tham số**.
3. **Stage 2 (`layer2`):** 2 khối `BasicBlock` liên tiếp (128 kênh, stride 2 ở khối đầu) $\to$ **525.568 tham số**.
4. **Stage 3 (`layer3`):** 2 khối `BasicBlock` liên tiếp (256 kênh, stride 2 ở khối đầu) $\to$ **2.099.712 tham số**.
5. **Stage 4 (`layer4`):** 2 khối `BasicBlock` liên tiếp (512 kênh, stride 2 ở khối đầu) $\to$ **8.393.728 tham số**.
6. **Global Pooling:** `avgpool` (`AdaptiveAvgPool2d((1, 1))`) $\to$ Trích xuất feature vector 512 chiều.

#### DenseNet-121 Backbone:
1. **Stem:** `conv0` ($7 \times 7$, stride 2, padding 3, 64 filters) + `norm0` + `relu0` + `pool0` ($3 \times 3$, stride 2, padding 1) $\to$ **9.536 tham số**.
2. **Stage 1:** `denseblock1` (6 dense layers) + `transition1` (`norm` + `relu` + `conv` $1 \times 1$ + `avgpool` $2 \times 2$, 128 channels) $\to$ **368.320 tham số**.
3. **Stage 2:** `denseblock2` (12 dense layers) + `transition2` (conv $1 \times 1$ + `avgpool` $2 \times 2$, 256 channels) $\to$ **1.051.776 tham số**.
4. **Stage 3:** `denseblock3` (24 dense layers) + `transition3` (conv $1 \times 1$ + `avgpool` $2 \times 2$, 512 channels) $\to$ **3.364.096 tham số**.
5. **Stage 4:** `denseblock4` (16 dense layers, trích xuất 1024 channels) $\to$ **2.158.080 tham số**.
6. **Pre-classifier Norm:** `norm5` (`BatchNorm2d(1024)`) + `relu` + `AdaptiveAvgPool2d((1, 1))` $\to$ **2.048 tham số**, trích xuất feature vector 1.024 chiều.

### 5.3. Thành phần chính (Main Components)
- **Tầng nông (Low-level Feature Extractors):** Stem và Stages 1–3 chịu trách nhiệm trích xuất các đặc trưng cơ bản (đường nét, góc cạnh, dải màu, kết cấu thô). Trong cả hai chiến lược, các tầng này được **đóng băng tuyệt đối** để bảo toàn tri thức nền tảng đã học từ ImageNet.
- **Tầng sâu (High-level Semantic Extractors):** Stage 4 (`layer4` trên ResNet-18 hoặc `denseblock4` + `norm5` trên DenseNet-121) tổng hợp các hình thái ngữ nghĩa trừu tượng mang tính chuyên biệt của vật thể. Trong Fine-Tuning, tầng này được **mở khóa có kiểm soát** bằng tốc độ học nhỏ ($\eta = 0.0001$) để thích nghi với các chi tiết đặc thù của 10 lớp CIFAR-10.
- **Đầu phân loại (Classification / Prediction Head):**
  - *ResNet-18 Head:* `model.fc = nn.Linear(in_features=512, out_features=10)` ($512 \times 10 + 10 = 5.130$ tham số).
  - *DenseNet-121 Head:* `model.classifier = nn.Linear(in_features=1024, out_features=10)` ($1024 \times 10 + 10 = 10.250$ tham số).

### 5.4. Thống kê tham số mô hình (Model Parameters Profiling)

| Cấu hình Thí nghiệm | Tổng tham số | Tham số Huấn luyện (Trainable) | Tham số Đóng băng (Frozen) | Tỷ lệ Trainable |
|:---|:---:|:---:|:---:|:---:|
| **`resnet18_transfer`** | 11.181.642 | **5.130** | 11.176.512 | **0,05%** |
| **`resnet18_finetune`** | 11.181.642 | **8.398.858** | 2.782.784 | **75,11%** |
| **`densenet121_transfer`** | 6.964.106 | **10.250** | 6.953.856 | **0,15%** |
| **`densenet121_finetune`** | 6.964.106 | **2.170.378** | 4.793.728 | **31,17%** |

### 5.5. Phân tích hành vi BatchNorm (Parameters vs. Buffers)
- **Bản chất kỹ thuật:** Trong PyTorch, các tầng `BatchNorm2d` chứa hai thành phần phân biệt rõ rệt:
  1. *Parameters:* `weight` ($\gamma$) và `bias` ($\beta$). Thuộc tính `requires_grad = False` chỉ ngăn cản việc tính toán gradient cho $\gamma$ và $\beta$.
  2. *Buffers:* `running_mean`, `running_var` và `num_batches_tracked`. Đây là các biến trạng thái tích lũy, không tham gia vào đồ thị gradient (Autograd Graph).
- **Hành vi khi huấn luyện:**
  - Ở chế độ `model.train()`: Mặc dù `requires_grad = False`, các buffer thống kê vẫn tiếp tục cập nhật theo công thức chạy tích lũy:
    $$\mu_{\text{run}} = (1 - \text{momentum}) \cdot \mu_{\text{run}} + \text{momentum} \cdot \mu_{\text{batch}}$$
  - Ở chế độ `model.eval()`: 100% buffer bị khóa tĩnh hoàn toàn; mạng sử dụng cố định các giá trị `running_mean` và `running_var` đã được tích lũy.
- **Kiểm chứng thực nghiệm:** Xác thực độc lập qua [`scripts/check_frozen_bn.py`](../scripts/check_frozen_bn.py), bảo đảm không xảy ra rò rỉ gradient hay sai lệch trạng thái bộ đệm.

---

## 6. MODEL INPUT / OUTPUT

### 6.1. Input Specification
- **Data Type:** `torch.float32` (Tensor số thực dấu phẩy động 32-bit).
- **Tensor Shape:** `[B, 3, 128, 128]`
  - $B$: Kích thước lô (Batch Size $B \in \{16, 32\}$).
  - $C = 3$: Kênh màu RGB (Red, Green, Blue).
  - $H = 128, W = 128$: Kích thước không gian ảnh sau khi tiền xử lý.
- **Pipeline Tiền xử lý (Preprocessing Pipeline):**
  1. *Spatial Resizing:* Ảnh gốc CIFAR-10 $32 \times 32$ được phóng to lên kích thước chuẩn $128 \times 128$ qua phép nội suy song tuyến tính (`InterpolationMode.BILINEAR`). Thao tác này tương thích với trường tiếp nhận (Receptive Field) của 5 tầng giảm mẫu ($2^5 = 32$), giúp giữ lại bản đồ đặc trưng $4 \times 4$ trước Global Average Pooling.
  2. *Data Augmentation (chỉ kích hoạt trên tập Train):*
     - `transforms.RandomCrop(128, padding=16, padding_mode="reflect")`: Tạo ra sự xê dịch vị trí ngẫu nhiên, tăng tính bền vững không gian.
     - `transforms.RandomHorizontalFlip(p=0.5)`: Lật ảnh ngẫu nhiên theo trục ngang, mô phỏng đối xứng vật lý.
  3. *Tensor Conversion (`ToTensor`):* Chuyển đổi dữ liệu từ mảng ảnh `uint8 [0, 255]` sang tensor `float32 [0.0, 1.0]`, đồng thời hoán vị trục từ `[H, W, C]` sang `[C, H, W]`.
  4. *ImageNet Normalization:* Chuẩn hóa độc lập theo từng kênh màu dựa trên phân phối ImageNet:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
     $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

### 6.2. Output Specification
- **Output Type:** `torch.float32` (Tensor Logits thô chưa qua kích hoạt Softmax).
- **Tensor Shape:** `[B, 10]` ($B$: Batch Size, 10 chiều tương ứng 10 lớp phân loại của CIFAR-10).
- **Interpretation & Post-processing:**
  - *Logits:* Vector điểm số thô $\mathbf{z} = [z_0, z_1, \dots, z_9] \in \mathbb{R}^{10}$ cho 10 danh mục:
    `[0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck]`.
  - *Xác suất dự đoán (Softmax Probabilities):*
    $$P(y = c \mid \mathbf{x}) = \frac{\exp(z_c)}{\sum_{j=0}^{9} \exp(z_j)}, \quad \sum_{c=0}^9 P(y = c \mid \mathbf{x}) = 1.0$$
  - *Nhãn dự đoán cuối cùng (Predicted Class):*
    $$\hat{y} = \arg\max_{c \in \{0, \dots, 9\}} z_c$$
  - *Hàm mất mát huấn luyện (Training Loss):*
    $$\mathcal{L}_{CE} = -\frac{1}{B} \sum_{i=1}^B \sum_{c=0}^9 y_{i, c} \log\left(\frac{\exp(z_{i, c})}{\sum_{j=0}^9 \exp(z_{i, j})}\right)$$
    Triển khai qua `torch.nn.CrossEntropyLoss()` kết hợp `log_softmax` và `nll_loss` để triệt tiêu nguy cơ tràn số học (numerical underflow/overflow).

---

## 7. SOURCE FILES & RESPONSIBILITIES

| Đường dẫn tệp mã nguồn | Trách nhiệm kỹ thuật trong hệ thống |
|:---|:---|
| [`src/lab2/models.py`](../src/lab2/models.py) | Định nghĩa hàm khởi tạo `build_resnet18()`, `build_densenet121()`, `build_model()`; cơ chế đóng băng/mở khóa tham số `requires_grad`; thay thế Classifier Head 10 lớp; và hàm đếm tham số `count_parameters()`. |
| [`scripts/inspect_models.py`](../scripts/inspect_models.py) | Công cụ CLI phục vụ **Bước 3**: In kiến trúc mạng PyTorch (`--print-model`), phân tích bóc tách các khối tầng (`--layers`), khảo sát chi tiết tensor tham số và gradient (`--params`), và xuất bảng tổng hợp tham số. |
| [`configs/config.yaml`](../configs/config.yaml) | Quản lý cấu hình tập trung cho toàn bộ 6 thí nghiệm (`resnet18_transfer`, `resnet18_finetune`, `densenet121_transfer`, `densenet121_finetune`, `resnet18_transfer_lr1e4`, `resnet18_transfer_bs32`). |
| [`src/lab2/train.py`](../src/lab2/train.py) | Triển khai vòng lặp huấn luyện chu trình 6 bước chuẩn, phân tách Differential Learning Rate (`create_optimizer`), giám sát Loss/Accuracy, và kiểm tra bẫy NaN/Inf loss. |
| [`src/lab2/checkpoint.py`](../src/lab2/checkpoint.py) | Quản lý lưu trữ trọng số mô hình nguyên tử an toàn (`atomic_save` qua tệp tạm `.tmp`), kiểm toán tính toàn vẹn checkpoint, phân tách `_best.pth`, `_latest.pth` và cắt tỉa checkpoint cũ. |
| [`src/lab2/evaluate.py`](../src/lab2/evaluate.py) | Engine đánh giá checkpoint độc lập trên tập Test 10.000 ảnh; tính toán Loss, Top-1 Accuracy, Per-class Accuracy, và sinh Ma trận nhầm lẫn (Confusion Matrix). |
| [`scripts/check_frozen_bn.py`](../scripts/check_frozen_bn.py) | Kịch bản kiểm chứng độc lập hành vi cập nhật bộ đệm thống kê (`running_mean`, `running_var`) của các tầng BatchNorm trong backbone khi chạy ở chế độ `train` và `eval`. |
| [`tests/test_models.py`](../tests/test_models.py) | Bộ kiểm thử tự động kiểm tra cơ chế đóng băng trọng số, đếm tham số trainable/frozen, và kích thước tensor đầu ra của các mô hình. |

---

## 8. VERIFICATION & REPRODUCIBILITY AUDIT

### 8.1. Kiểm chứng cấu trúc tầng qua CLI
```bash
# 1. In toàn bộ cấu trúc phân tầng PyTorch ra màn hình
py scripts/inspect_models.py --print-model all

# 2. Phân tích chi tiết từng khối tầng và trạng thái Trainable/Frozen
py scripts/inspect_models.py --layers all --method finetune

# 3. Liệt kê toàn bộ tensor tham số, kích thước shape và requires_grad
py scripts/inspect_models.py --params resnet18 --method transfer
```

### 8.2. Kiểm thử tự động (Unit Test Suite)
Bộ kiểm thử tự động tại [`tests/test_models.py`](../tests/test_models.py) và toàn bộ dự án đã được thực thi và vượt qua:
- **Kết quả kiểm thử:** `53 passed in 12.23s` (`py -m pytest -q`).
- **Nội dung kiểm thực:**
  - `test_build_model_resnet18_transfer`: Xác thực 5.130 tham số trainable và output shape `[B, 10]`.
  - `test_build_model_resnet18_finetune`: Xác thực 8.398.858 tham số trainable (`layer4` + `fc`).
  - `test_build_model_densenet121_transfer`: Xác thực 10.250 tham số trainable và output shape `[B, 10]`.
  - `test_build_model_densenet121_finetune`: Xác thực 2.170.378 tham số trainable (`denseblock4` + `norm5` + `classifier`).
  - `test_count_parameters`: Kiểm tra tính toàn vẹn của thuật toán đếm tham số trên tensor PyTorch.

---

*Tài liệu được bảo toàn 100% thông tin kỹ thuật, lưu trữ chính thức tại [`docs/model.md`](model.md) và [`docs/MODEL_DOCUMENTATION.md`](MODEL_DOCUMENTATION.md).*
