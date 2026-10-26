# Experiment Plan — Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning

## 1. Mục đích thực nghiệm (Experiment Goals)
Kế hoạch thực nghiệm này được thiết kế nhằm so sánh có hệ thống và công bằng (fair comparison) giữa hai họ kiến trúc mạng sâu kinh điển (**ResNet18** và **DenseNet121**) dưới hai phương thức thích nghi mô hình (**Transfer Learning** và **Fine-Tuning**) trên tập dữ liệu phân loại ảnh **CIFAR-10**.

## 2. Ma trận 4 Thí nghiệm Bắt buộc (Four Core Experiments)

| Experiment ID | Kiến trúc | Phương pháp | Các tầng được huấn luyện (Trainable Layers) | Các tầng bị đóng băng (Frozen Layers) | Optimizer & Learning Rate | Batch Size | Epochs |
|:---|:---:|:---:|:---|:---|:---|:---:|:---:|
| `resnet18_transfer` | ResNet18 | Transfer Learning | Duy nhất `model.fc` | `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4` | Adam (`lr=1e-3`, `weight_decay=1e-4`) | 16 | 5 |
| `resnet18_finetune` | ResNet18 | Fine-Tuning | `model.layer4` + `model.fc` | `conv1`, `bn1`, `layer1`, `layer2`, `layer3` | Adam (`backbone_lr=1e-4`, `fc_lr=1e-3`) | 16 | 5 |
| `densenet121_transfer` | DenseNet121 | Transfer Learning | Duy nhất `model.classifier` | Toàn bộ `model.features` (`conv0`..`norm5`) | Adam (`lr=1e-3`, `weight_decay=1e-4`) | 16 | 5 |
| `densenet121_finetune` | DenseNet121 | Fine-Tuning | `denseblock4` + `norm5` + `classifier` | `conv0` đến `denseblock3` | Adam (`backbone_lr=1e-4`, `classifier_lr=1e-3`) | 16 | 5 |

## 3. Các biến cần kiểm soát (Controlled Variables)
Để đảm bảo tính khoa học và công bằng giữa 4 thí nghiệm:
1. **Dữ liệu phân chia (Data Split):** Cả 4 thí nghiệm dùng chung 100% cùng quy tắc chia dữ liệu:
   - 45.000 mẫu train.
   - 5.000 mẫu validation (seed ngẫu nhiên cố định = 42).
   - 10.000 mẫu test chính thức.
2. **Quy tắc tiền xử lý (Preprocessing):**
   - Kích thước ảnh: $128 \times 128$ (hoặc $224 \times 224$).
   - Chuẩn hóa: ImageNet mean `[0.485, 0.456, 0.406]` và std `[0.229, 0.224, 0.225]`.
   - Augmentation: Giống nhau cho tập train (RandomHorizontalFlip, RandomCrop) và không có augmentation trên validation/test.
3. **Hàm mất mát (Loss Function):** Sử dụng `nn.CrossEntropyLoss()` cho toàn bộ các thí nghiệm.
4. **Quy tắc chọn Checkpoint:** Dựa trên cùng 1 thước đo duy nhất: `val_acc` cao nhất trong quá trình huấn luyện.
5. **Thiết bị phần cứng:** Chạy trên CPU (`device='cpu'`) để kiểm soát công bằng thời gian thực thi.

## 4. Biến độc lập (Independent Variables)
1. **Kiến trúc mô hình:**
   - ResNet18: Residual Connections (Skip Connections), 11.18 triệu tham số.
   - DenseNet121: Dense Connectivity (nối đặc trưng từ mọi tầng trước), 6.96 triệu tham số.
2. **Phương pháp thích nghi:**
   - Transfer Learning (Feature Extractor): Chỉ huấn luyện tầng phân loại mới.
   - Fine-Tuning: Huấn luyện tầng phân loại mới kết hợp tinh chỉnh các khối tích chập sâu nhất.

## 5. Tiêu chí so sánh và Biến phụ thuộc (Evaluation Criteria)
1. **Độ chính xác (Accuracy):**
   - Training Accuracy theo epoch.
   - Validation Accuracy theo epoch.
   - **Test Accuracy cuối cùng** trên tập kiểm thử độc lập (tiêu chí chính).
2. **Độ mất mát (Loss):**
   - Training Loss & Validation Loss qua từng epoch (phát hiện hiện tượng Underfitting hoặc Overfitting).
   - Test Loss.
3. **Độ chính xác theo từng lớp (Per-class Accuracy):**
   - Phân tích hiệu năng phân loại trên từng lớp trong 10 lớp của CIFAR-10.
4. **Ma trận nhầm lẫn (Confusion Matrix):**
   - Xác định những cặp lớp hay bị phân loại nhầm (ví dụ: Cat và Dog, Automobile và Truck).
5. **Hiệu suất tính toán:**
   - Số lượng tham số cần tối ưu (`trainable parameters`).
   - Thời gian huấn luyện trung bình mỗi epoch và tổng thời gian chạy.

## 6. Quy trình thực hiện (Execution Procedure)
1. **Bước 1 — Kiểm tra kiểm thử tự động (Unit Tests):** Chạy `pytest` đảm bảo logic đóng băng/mở khóa tham số của cả 4 cấu hình hoạt động đúng.
2. **Bước 2 — Chạy Smoke Test:** Chạy `python scripts/run_experiments.py --smoke-test` trên một mẫu nhỏ để xác thực toàn bộ luồng huấn luyện, checkpoint và TensorBoard logging.
3. **Bước 3 — Huấn luyện và Đánh giá 4 thí nghiệm:**
   - Chạy tuần tự 4 thí nghiệm qua `python scripts/run_experiments.py --all`.
   - Giám sát qua giao diện TensorBoard: `tensorboard --logdir=runs`.
4. **Bước 4 — Thu thập và Báo cáo Kết quả Thực tế:**
   - Tổng hợp các file metrics JSON và hình vẽ Confusion Matrix vào tài liệu `results.md`.
