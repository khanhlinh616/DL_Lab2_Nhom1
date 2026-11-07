# Model Documentation

## 1. Created Date
- 2026-10-27

## 2. Updated Date
- 2026-11-07

## 3. Purpose
- **Objective:** Nghiên cứu, thực nghiệm và so sánh đối chứng có hệ thống khả năng chuyển giao tri thức (Transfer Learning) từ các mô hình học sâu thị giác tiền huấn luyện trên ImageNet-1K sang bài toán phân loại ảnh CIFAR-10 trong môi trường CPU-First, giải quyết bài toán chống quá khớp (Overfitting) và tối ưu hóa tài nguyên tính toán.
- **Task:** Xây dựng, điều chỉnh hai kiến trúc ResNet-18 và DenseNet-121, so sánh 2 phương pháp thích nghi (Transfer Learning vs Fine-Tuning với Differential LR), khảo sát độ nhạy siêu tham số (LR $10^{-3}$ vs $10^{-4}$, Batch Size $16$ vs $32$), và hoàn thành Bước 3 Khám phá kiến trúc mô hình (in cấu trúc, xác định layers và tham số).
- **Expected Outcome:** Đạt độ chính xác kiểm thử Test Accuracy $> 90\%$ trên tập Test 10.000 ảnh độc lập khi Fine-Tuning, làm sáng tỏ sự đánh đổi giữa hiệu quả tham số và chi phí tính toán thực tế, đảm bảo Zero Data Leakage, lưu trữ checkpoint an toàn nguyên tử và vượt qua 100% bài kiểm thử tự động pytest.

## 4. Initial Model
- **Model Name:** ResNet-18 (He et al., CVPR 2016) và DenseNet-121 (Huang et al., CVPR 2017).
- **Pretrained Weights:** `ResNet18_Weights.IMAGENET1K_V1` (Top-1 Acc: 69,758%) và `DenseNet121_Weights.IMAGENET1K_V1` (Top-1 Acc: 74,434%).
- **Source:** Thư viện học sâu chuẩn `torchvision.models` (Torchvision 0.29.1+cpu).
- **Initialization Strategy:** Giữ nguyên trọng số pretrained ImageNet-1K V1 cho backbone; khởi tạo ngẫu nhiên tầng phân loại 10 lớp mới theo phân phối Kaiming Uniform (`weight` $\sim \mathcal{U}(-\sqrt{k}, \sqrt{k})$ với $k = \frac{1}{\text{in\_features}}$) và `bias` $\sim \mathcal{U}(-\sqrt{k}, \sqrt{k})$. Khóa backbone đối với Transfer Learning và mở khóa khối sâu nhất (`layer4` hoặc `denseblock4` + `norm5`) đối với Fine-Tuning.

## 5. Model Architecture
- **Architecture Overview:** ResNet-18 (Residual Networks với đường tắt cộng dồn $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ chống suy biến gradient) và DenseNet-121 (Densely Connected Networks với ghép nối kênh đặc trưng $\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \dots, \mathbf{x}_{\ell-1}])$ tối đa hóa tái sử dụng đặc trưng). Cả hai mô hình đại diện cho hai triết lý thiết kế mạng nơ-ron học sâu đối lập kinh điển, được lựa chọn nhờ sự tương thích tối ưu với quy mô dữ liệu CIFAR-10 ($32 \times 32$, 45k train) nhằm chống quá khớp (Overfitting), đồng thời làm rõ sự đánh đổi giữa số lượng tham số và chi phí tính toán thực tế. Quy trình Bước 3 (Explore the Model Architecture) được thực thi hoàn chỉnh qua `scripts/inspect_models.py` (`--print-model`, `--layers`, `--params`).
- **Backbone:**
  - *ResNet-18:* Input Stem (`conv1` $7 \times 7$ + `bn1` + `maxpool` $3 \times 3$, 9.536 params) kết hợp 4 khối dư phân cấp: `layer1` (2 BasicBlocks, 147.968 params), `layer2` (2 BasicBlocks, 525.568 params), `layer3` (2 BasicBlocks, 2.099.712 params), `layer4` (2 BasicBlocks, 8.393.728 params), và `avgpool` (`AdaptiveAvgPool2d((1, 1))`) xuất ra feature vector 512 chiều.
  - *DenseNet-121:* Input Stem (`conv0` $7 \times 7$ + `norm0` + `pool0`, 9.536 params) kết hợp 4 khối kết nối dày đặc và 3 tầng chuyển tiếp: `denseblock1` (6 dense layers) + `transition1` (368.320 params), `denseblock2` (12 dense layers) + `transition2` (1.051.776 params), `denseblock3` (24 dense layers) + `transition3` (3.364.096 params), `denseblock4` (16 dense layers, 2.158.080 params), và `norm5` (`BatchNorm2d(1024)` + `relu` + `AdaptiveAvgPool2d((1, 1))`, 2.048 params) xuất ra feature vector 1.024 chiều.
- **Main Components:**
  - *ResNet-18:* Tầng nông (`conv1`, `bn1`, `layer1..3`) trích xuất hoa văn, đường nét và kết cấu cục bộ; Tầng sâu (`layer4`) trích xuất ngữ nghĩa cấp cao; Tầng kết thúc (`avgpool` + `fc`) phân loại 10 lớp.
  - *DenseNet-121:* Tầng nông (`conv0`, `norm0`, `denseblock1..3`, `transition1..3`) trích xuất đặc trưng đa tầng với hệ số tăng trưởng $k=32$; Tầng sâu (`denseblock4`, `norm5`) tổng hợp đặc trưng ngữ nghĩa trừu tượng; Tầng kết thúc (`classifier`) phân loại 10 lớp.
- **Classification / Prediction Head:**
  - *ResNet-18 Head:* `model.fc = nn.Linear(in_features=512, out_features=10)` ($512 \times 10 + 10 = 5.130$ tham số).
  - *DenseNet-121 Head:* `model.classifier = nn.Linear(in_features=1024, out_features=10)` ($1024 \times 10 + 10 = 10.250$ tham số).
- **Trainable Parameters:**
  - `resnet18_transfer`: **5.130** tham số trainable (0,05%) / 11.176.512 tham số frozen / 11.181.642 tổng.
  - `resnet18_finetune`: **8.398.858** tham số trainable (75,11% - mở khóa `layer4` và `fc`) / 2.782.784 tham số frozen / 11.181.642 tổng.
  - `densenet121_transfer`: **10.250** tham số trainable (0,15%) / 6.953.856 tham số frozen / 6.964.106 tổng.
  - `densenet121_finetune`: **2.170.378** tham số trainable (31,17% - mở khóa `denseblock4`, `norm5`, `classifier`) / 4.793.728 tham số frozen / 6.964.106 tổng.
  - *Kiểm toán hành vi BatchNorm buffer:* `requires_grad=False` chỉ khóa trọng số tự do; ở `train()` mode, 100% buffer running stats trong backbone vẫn cập nhật theo batch; ở `eval()` mode, 100% buffer hoàn toàn đóng băng tĩnh (xác thực qua [`scripts/check_frozen_bn.py`](../scripts/check_frozen_bn.py)).

## 6. Model Input / Output

### Input
- **Data Type:** `torch.float32` (Tensor dấu phẩy động 32-bit).
- **Shape:** `[B, 3, 128, 128]` (Batch size $B \in \{16, 32\}$, 3 kênh màu RGB, chiều cao $H=128$, chiều rộng $W=128$).
- **Preprocessing:**
  1. *Spatial Resizing:* Ảnh gốc CIFAR-10 $32 \times 32$ pixels được phóng đại lên $128 \times 128$ qua nội suy song tuyến (Bilinear Interpolation) để tương thích trường tiếp nhận (Receptive Field) của backbone ImageNet.
  2. *Data Augmentation (chỉ trên tập Train):* `RandomCrop(128, padding=16, padding_mode='reflect')` và `RandomHorizontalFlip(p=0.5)` chống quá khớp.
  3. *Tensor Conversion:* `ToTensor()` chuyển đổi mảng ảnh `uint8 [0, 255]` sang `float32 [0.0, 1.0]`, chuyển trục không gian từ `[H, W, C]` sang `[C, H, W]`.
  4. *ImageNet Normalization:* Chuẩn hóa độc lập theo từng kênh màu dựa trên phân phối ImageNet: Mean $\mu = [0.485, 0.456, 0.406]$, Std $\sigma = [0.229, 0.224, 0.225]$, công thức $x_{norm} = \frac{x - \mu}{\sigma}$.

### Output
- **Output Type:** `torch.float32` (Tensor Logits chưa qua chuẩn hóa xác suất).
- **Shape:** `[B, 10]` (Batch size $B$, 10 chiều logits tương ứng 10 lớp đối tượng đích của CIFAR-10).
- **Interpretation:**
  - *Logits:* Điểm số thô $\mathbf{z} \in \mathbb{R}^{10}$ cho từng lớp đối tượng: `[0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck]`.
  - *Xác suất dự đoán:* Áp dụng Softmax: $P(y = c \mid \mathbf{x}) = \frac{\exp(z_c)}{\sum_{j=1}^{10} \exp(z_j)}$, với $\sum_{c=1}^{10} P(y=c) = 1.0$.
  - *Dự đoán nhãn (Predicted Class):* $\hat{y} = \arg\max_{c \in \{0, \dots, 9\}} z_c$.
  - *Hàm mất mát (Training Loss):* `nn.CrossEntropyLoss()` kết hợp `log_softmax` và `NLLLoss` chống tràn số.

## 7. Source Files
| File | Responsibility |
|---|---|
| [`src/lab2/models.py`](../src/lab2/models.py) | Định nghĩa hàm khởi tạo `build_resnet18()`, `build_densenet121()`, `build_model()`, cơ chế đóng băng/mở khóa tham số `requires_grad`, thay thế Classifier Head 10 lớp, và hàm đếm tham số `count_parameters()`. |
| [`scripts/inspect_models.py`](../scripts/inspect_models.py) | Công cụ CLI phục vụ Bước 3: In kiến trúc mạng PyTorch (`--print-model`), phân tích bóc tách các khối tầng (`--layers`), khảo sát chi tiết tensor tham số và gradient (`--params`), và xuất bảng tổng hợp. |
| [`configs/config.yaml`](../configs/config.yaml) | Quản lý cấu hình tập trung cho toàn bộ 6 thí nghiệm (`resnet18_transfer`, `resnet18_finetune`, `densenet121_transfer`, `densenet121_finetune`, `lr1e4`, `bs32`). |
| [`src/lab2/train.py`](../src/lab2/train.py) | Vòng lặp huấn luyện chu trình 6 bước chuẩn, phân tách Differential Learning Rate (`create_optimizer`), giám sát Loss/Accuracy, và kiểm tra NaN/Inf loss. |
| [`src/lab2/checkpoint.py`](../src/lab2/checkpoint.py) | Quản lý lưu trữ trọng số mô hình nguyên tử an toàn (`atomic_save` qua `.tmp`), kiểm toán tính toàn vẹn checkpoint, phân tách `_best.pth`, `_latest.pth` và cắt tỉa checkpoint cũ. |
| [`src/lab2/evaluate.py`](../src/lab2/evaluate.py) | Engine đánh giá checkpoint độc lập trên tập Test 10.000 ảnh, tính toán Loss, Top-1 Accuracy, Per-class Accuracy, và sinh Ma trận nhầm lẫn (Confusion Matrix). |
| [`scripts/check_frozen_bn.py`](../scripts/check_frozen_bn.py) | Kịch bản kiểm chứng độc lập hành vi cập nhật bộ đệm thống kê (`running_mean`, `running_var`) của các tầng BatchNorm trong backbone khi chạy ở chế độ `train` và `eval`. |
| [`tests/test_models.py`](../tests/test_models.py) | Bộ kiểm thử tự động kiểm tra cơ chế đóng băng trọng số, đếm tham số trainable/frozen, và kích thước tensor đầu ra của các mô hình. |
