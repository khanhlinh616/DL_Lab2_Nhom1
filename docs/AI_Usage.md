# AI Usage Report — Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning

## 1. Công cụ AI đã sử dụng (Tools Used)
- **Công cụ chính:** Antigravity AI Assistant (Gemini 3.8 Flash).
- **Hệ điều hành môi trường:** Windows 11 64-bit (Intel Core i5-12450HX, 24 GB RAM, CPU-first).

## 2. Mục đích chính (Main Purposes)
- Thiết kế kiến trúc mã nguồn modular theo chuẩn công nghiệp (tương đồng mẫu tham khảo `mrdbourke/pytorch-deep-learning`).
- Triển khai pipeline huấn luyện, kiểm thực cấu hình, lưu checkpoint, TensorBoard logging và đánh giá kiểm thử theo các slide bài giảng Chương 3 (PhD. Nguyễn Thị Khánh Tiên, ĐH Giao thông Vận tải TP.HCM).
- Viết bộ kiểm thử đơn vị tự động (Unit & Integration Tests) với Pytest nhằm kiểm chứng độc lập tính đóng băng (freezing) và mở khóa (unfreezing) của các lớp trong 4 thí nghiệm.
- Hỗ trợ gỡ lỗi kỹ thuật môi trường (chứng chỉ SSL khi tải CIFAR-10 trên Windows, tối ưu hóa tốc độ tải dữ liệu và cấu hình CPU-first).

## 3. Đóng góp của sinh viên / người dùng (Student Contributions)
- Đưa ra yêu cầu chi tiết (Master Build Prompt) dựa trên đề cương Practice 2 và Slide bài giảng Chương 3.
- Rà soát, đối chiếu và phê duyệt bản thiết kế kỹ thuật (Phase 2 — Design) trước khi tiến hành sinh mã nguồn.
- Xác định phạm vi phần cứng thực tế (CPU Intel Core i5, không GPU NVIDIA) để định hướng thiết kế CPU-first và cờ `--smoke-test`.
- Đánh giá tính chính xác của các kết quả thực nghiệm sau khi chạy.

## 4. Các thành phần quan trọng do AI sinh ra (Important AI-Generated Components)
- `src/lab2/models.py`: Cơ chế xây dựng mô hình ResNet18 và DenseNet121, tính toán số lượng tham số trainable/frozen và mở khóa chính xác các tầng (`layer4` cho ResNet18; `denseblock4` + `norm5` cho DenseNet121).
- `src/lab2/train.py`: Vòng lặp huấn luyện và validation tuân thủ nghiêm ngặt 6 bước của Slide 8-9 và cơ chế đánh giá `model.eval()`, `torch.no_grad()` của Slide 10-11, hỗ trợ differential learning rates.
- `src/lab2/tensorboard_logger.py`: Wrapper quản lý `SummaryWriter`, chuẩn hóa scalar logging, model graph và image grid.
- `src/lab2/checkpoint.py`: Cơ chế lưu và nạp checkpoint toàn diện với siêu dữ liệu.
- `tests/`: Bộ 14 unit tests kiểm tra tính cô lập của dữ liệu, shape tensor, luồng gradient và tính toàn vẹn của checkpoint.

## 5. Mục đích của câu lệnh điều khiển (Prompt Purpose)
- Thiết lập quy trình làm việc nghiêm ngặt theo 9 giai đoạn (Describe $\to$ Design $\to$ Generate $\to$ Review $\to$ Run $\to$ Dry-run $\to$ Test $\to$ Refactor $\to$ Commit).
- Áp dụng nguyên tắc "Source-of-Truth" tuyệt đối: căn cứ vào đề bài và slide bài giảng để xác định yêu cầu bắt buộc, không đưa ra số liệu giả định chưa được kiểm chứng.

## 6. Chỉnh sửa và Tinh chỉnh của Sinh viên (Student Modification)
- Điều chỉnh kích thước ảnh mặc định ($128 \times 128$) trong cấu hình để cân bằng giữa độ chính xác và thời gian huấn luyện trên CPU.
- Thêm cờ `--smoke-test` để kiểm tra luồng hoạt động nhanh mà không làm treo máy tính cá nhân.
- Xử lý các đường dẫn tệp tin và thiết lập `num_workers=0` để tương thích tối đa với hệ điều hành Windows.

## 7. Phương pháp Kiểm chứng (Validation)
- Chạy bộ kiểm thử tự động `pytest -v` (14/14 tests passed).
- Chạy script kiểm tra kiến trúc `inspect_models.py` để xác thực số lượng tham số đóng băng/mở khóa thực tế.
- Chạy smoke test thực nghiệm trên dữ liệu thật để kiểm tra lưu checkpoint và ghi nhận log TensorBoard.
