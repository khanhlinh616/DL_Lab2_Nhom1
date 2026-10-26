# Experimental Results & Analysis — Lab 2

## 1. Giới thiệu và Nguyên tắc Ghi nhận Kết quả
Báo cáo này tuân thủ nguyên tắc **Source-of-Truth**: Chỉ ghi nhận các kết quả đã thực sự được thực thi, kiểm tra và xác thực trên máy phát triển thực tế (Intel Core i5-12450HX, 24 GB RAM, Windows 11).
Tuyệt đối không bịa đặt số liệu hoặc giả định kết quả khi chưa thực nghiệm.

---

## 2. Kết quả Khảo sát Kiến trúc và Phân tích Tham số (Architecture Inspection)
Được đo trực tiếp thông qua lệnh `python scripts/inspect_models.py`:

| Experiment ID | Kiến trúc | Phương pháp | Tổng số tham số (Total) | Tham số Trainable | Tham số Bị đóng băng (Frozen) | Tỷ lệ Trainable (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `resnet18_transfer` | ResNet18 | Transfer Learning | 11,181,642 | 5,130 | 11,176,512 | 0.05% |
| `resnet18_finetune` | ResNet18 | Fine-Tuning | 11,181,642 | 8,398,858 | 2,782,784 | 75.11% |
| `densenet121_transfer` | DenseNet121 | Transfer Learning | 6,964,106 | 10,250 | 6,953,856 | 0.15% |
| `densenet121_finetune` | DenseNet121 | Fine-Tuning | 6,964,106 | 2,170,378 | 4,793,728 | 31.17% |

### Nhận xét phân tích kỹ thuật:
1. **ResNet18 Transfer Learning:** Chỉ duy nhất tầng kết nối đầy đủ cuối cùng (`fc.weight` và `fc.bias`) với $512 \times 10 + 10 = 5.130$ tham số được cập nhật. Toàn bộ các khối `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4` đều đóng băng hoàn toàn.
2. **ResNet18 Fine-Tuning:** Khi mở khóa khối tích chập sâu nhất `layer4` (gồm 2 BasicBlocks) cùng `fc`, số lượng tham số trainable tăng vọt lên 8.398.858 (chiếm 75.11% tổng mô hình), cho phép mô hình thích nghi các bộ lọc biểu diễn đặc trưng cấp cao theo dữ liệu CIFAR-10.
3. **DenseNet121 Transfer Learning:** Tầng phân loại `classifier` mới nhận đầu vào $1.024$ kênh, tạo ra $1.024 \times 10 + 10 = 10.250$ tham số cần huấn luyện (chiếm 0.15%).
4. **DenseNet121 Fine-Tuning:** Khác với ResNet, cấu trúc DenseNet có cơ chế nối ghép đặc trưng (feature concatenation). Việc mở khóa `denseblock4` (16 dense layers), `norm5` và `classifier` chỉ kích hoạt 2.170.378 tham số (chiếm 31.17%), giúp fine-tuning rất nhẹ nhàng và tiết kiệm tài nguyên tính toán hơn so với ResNet18.

---

## 3. Kết quả Smoke Test Xác thực Pipeline (Dry-run Verification)
Đã thực thi thành công chế độ `--smoke-test` trên tập mẫu thực tế để kiểm tra tính toàn vẹn của:
- Pipeline tiền xử lý CIFAR-10 (Resize, Normalize).
- Vòng lặp huấn luyện (train loop) & validation loop với `torch.no_grad()`.
- Ghi nhận metrics sang TensorBoard (`runs/`).
- Tự động lưu checkpoint tốt nhất (`checkpoints/resnet18_transfer_best.pth`).
- Nạp lại checkpoint để đánh giá tập test độc lập, xuất JSON metrics và vẽ Confusion Matrix (`outputs/`).

---

## 4. Bảng Kết quả Thực nghiệm Chính thức (Official Experiments Status)

| Experiment ID | Epochs | Batch Size | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Test Loss | Test Acc (%) | Trạng thái |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `resnet18_transfer` | 5 | 16 | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | Sẵn sàng thực thi |
| `resnet18_finetune` | 5 | 16 | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | Sẵn sàng thực thi |
| `densenet121_transfer` | 5 | 16 | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | Sẵn sàng thực thi |
| `densenet121_finetune` | 5 | 16 | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | *Chờ chạy full* | Sẵn sàng thực thi |

> *Ghi chú trung thực:* Bảng trên ghi nhận trạng thái sẵn sàng của 4 thí nghiệm. Khi người dùng ra lệnh huấn luyện chính thức (chạy full 5 epochs cho từng cấu hình hoặc toàn bộ qua `python scripts/run_experiments.py --all`), số liệu định lượng chính thức từ các file `outputs/metrics/*_test_metrics.json` sẽ được cập nhật vào bảng này.

---

## 5. Giới hạn hiện tại (Current Limitations)
1. **Ràng buộc phần cứng CPU:** Máy tính phát triển sử dụng CPU Intel Core i5-12450HX không có GPU NVIDIA CUDA. Việc huấn luyện 5 epochs trên toàn bộ 45.000 ảnh CIFAR-10 với ảnh resize 128x128 hoặc 224x224 sẽ tốn khoảng 30 - 60 phút cho mỗi thí nghiệm.
2. **Kích thước độ phân giải:** Ảnh CIFAR-10 gốc là 32x32. Việc phóng đại lên 128x128 (để tương thích receptive field của pretrained weights ImageNet) có thể làm mờ một số đặc trưng biên chi tiết so với ảnh gốc có độ phân giải tự nhiên cao.

---

## 6. Hướng phát triển và Cải tiến (Future Improvements)
1. **Triển khai Mixed Precision Training:** Tận dụng `torch.amp.autocast('cpu', dtype=torch.bfloat16)` nếu phiên bản PyTorch hỗ trợ để tăng tốc độ suy luận và huấn luyện trên CPU hiện đại.
2. **Learning Rate Scheduler:** Bổ sung `CosineAnnealingLR` hoặc `ReduceLROnPlateau` để tự động điều chỉnh tốc độ học khi độ mất mát trên tập validation bão hòa.
3. **Data Augmentation nâng cao:** Thử nghiệm bổ sung AutoAugment hoặc RandAugment cho giai đoạn Fine-tuning để giảm bớt nguy cơ quá khớp khi mở khóa nhiều tham số.
