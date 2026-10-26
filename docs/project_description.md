# Project Description — Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning

## 1. Giới thiệu bài toán (Problem Statement)
Trong thị giác máy tính, việc huấn luyện các mạng nơ-ron tích chập (Convolutional Neural Networks - CNNs) từ đầu (from scratch) trên các tập dữ liệu nhỏ thường gặp nhiều khó khăn:
- Nguy cơ quá khớp (overfitting) rất cao do số lượng tham số lớn so với kích thước tập dữ liệu.
- Đòi hỏi tài nguyên tính toán phần cứng lớn và thời gian huấn luyện kéo dài.
- Khó hội tụ và dễ bị hiện tượng triệt tiêu gradient (vanishing gradient).

Để khắc phục các thách thức trên, kỹ thuật **Transfer Learning (Học chuyển giao)** và **Fine-Tuning (Tinh chỉnh tham số)** cho phép tái sử dụng các tri thức (biểu diễn đặc trưng không gian sâu) từ các mô hình đã được huấn luyện trước (Pretrained Models) trên các tập dữ liệu quy mô lớn (như ImageNet - 1.2 triệu ảnh, 1.000 lớp) để áp dụng vào bài toán phân loại ảnh trên tập dữ liệu mục tiêu **CIFAR-10**.

## 2. Mục tiêu dự án (Objectives)
- **Tiếp cận và khai thác Pretrained Models:** Sử dụng chính thống các pretrained weights từ `torchvision.models` (ResNet18 và DenseNet121).
- **So sánh 2 chiến lược thích nghi mô hình:**
  1. *Transfer Learning (Feature Extraction):* Đóng băng hoàn toàn backbone trích xuất đặc trưng, chỉ huấn luyện classifier mới cho 10 lớp CIFAR-10.
  2. *Fine-Tuning:* Mở khóa các tầng tích chập sâu nhất của backbone (layer4 của ResNet18; denseblock4 và norm5 của DenseNet121) cùng với classifier mới để tối ưu hóa đồng thời với learning rate phù hợp.
- **Tuân thủ quy trình khoa học (Scientific & Test-Driven Workflow):**
  - Đảm bảo chia tập dữ liệu tái lập (reproducible split), tuyệt đối không để rò rỉ dữ liệu (data leakage) giữa tập train, validation và test.
  - Sử dụng tiêu chí validation metric để lựa chọn checkpoint tốt nhất.
  - Chỉ đánh giá tập Test sau khi đã khóa checkpoint tốt nhất.
  - Giám sát toàn bộ tiến trình huấn luyện qua TensorBoard theo đúng nội dung Chương 3 (Training Deep Neural Networks).

## 3. Phạm vi dự án (Scope)
- **Tập dữ liệu:** CIFAR-10 (60.000 ảnh màu 32x32, 10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
  - Train: 45.000 ảnh.
  - Validation: 5.000 ảnh (tách từ train chính thức bằng seed cố định).
  - Test: 10.000 ảnh (giữ nguyên tập test chính thức).
- **Kiến trúc mô hình:**
  - `ResNet18` (Residual Networks - 18 layers).
  - `DenseNet121` (Densely Connected Convolutional Networks - 121 layers).
- **Môi trường phần cứng:** Tối ưu hóa CPU-first (Intel Core i5-12450HX, 24 GB RAM, Windows 11).

## 4. Đầu ra dự kiến (Expected Deliverables)
1. **Mã nguồn modular hoàn chỉnh:**
   - Cấu trúc thư mục chuẩn mực (`src/lab2/`, `scripts/`, `configs/`, `tests/`).
   - Module nạp cấu hình, chuẩn bị dữ liệu, xây dựng mô hình, vòng lặp train/val, checkpoint, TensorBoard và đánh giá.
2. **Bộ kiểm thử tự động (Unit & Integration Tests):**
   - Đảm bảo 100% test pass với pytest, kiểm chứng độc lập tính đóng băng/mở khóa tham số và luồng tính toán gradient.
3. **Checkpoints & Logs:**
   - Các file checkpoint tốt nhất (`*_best.pth`) và mới nhất (`*_latest.pth`).
   - TensorBoard tfevent logs theo dõi đường cong Loss và Accuracy.
4. **Báo cáo định lượng:**
   - Kết quả đánh giá trên tập test (Test Loss, Test Accuracy, Per-class Accuracy, Confusion Matrix).
