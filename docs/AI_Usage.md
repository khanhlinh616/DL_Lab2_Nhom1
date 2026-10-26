# AI Usage Report — Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning

## 1. Công cụ AI đã sử dụng (Tools Used)
- **Claude 3.7 Sonnet (Anthropic):** Được sử dụng để phân tích đề cương thực hành Practice 2 (Slide 23–24, Chương 3), soạn thảo tài liệu thiết kế kỹ thuật sơ bộ và xây dựng Master Prompt định hướng phát triển phần mềm theo chuẩn TDD & Modular Architecture.
- **Antigravity AI Assistant (Google DeepMind - Gemini 3.8 Flash):** Trợ lý lập trình AI chính trong môi trường Antigravity IDE, hỗ trợ thực thi trực tiếp các thao tác lập trình cặp (pair programming), chạy các lệnh shell kiểm thử, sinh mã nguồn, sinh notebook, điều tra hành vi BatchNorm và tự động hóa tài liệu.
- **Môi trường hệ thống:** Windows 11 64-bit (Intel Core i5-12450HX, 24 GB RAM, Python 3.14.7, CPU-first, không có GPU NVIDIA CUDA).

---

## 2. Mục đích chính (Main Purposes)
- Thiết kế kiến trúc mã nguồn modular chuẩn mực cho bài toán thị giác máy tính (`src/lab2/`, `scripts/`, `configs/`, `tests/`, `notebooks/`).
- Triển khai pipeline huấn luyện, kiểm thực cấu hình, lưu checkpoint, TensorBoard logging và đánh giá kiểm thử theo các slide bài giảng Chương 3 (PhD. Nguyễn Thị Khánh Tiên, ĐH Giao thông Vận tải TP.HCM).
- Xây dựng bộ kiểm thử tự động toàn diện theo phương pháp Test-Driven Development (TDD) với 51 unit & integration tests trên Pytest.
- Đo lường và điều tra thực nghiệm hành vi cập nhật bộ đệm `running_mean`, `running_var` của `BatchNorm2d` trên backbone bị đóng băng.
- Sinh hai notebook Jupyter tương tác theo chuẩn tự động hóa: notebook thực hành chính (`Lab2_Practice2_Main.ipynb`) và notebook phân tích kết quả chuyên sâu (`Lab2_Results_Analysis.ipynb`).
- Xử lý các lỗi môi trường đặc thù trên Windows (bảng mã `cp1252` gây `UnicodeEncodeError`, chứng chỉ SSL khi tải CIFAR-10).

---

## 3. Đóng góp của Sinh viên (Student Contributions)
- **Xác định mục tiêu & Ràng buộc:** Khảo sát kỹ đề bài thực hành, định hình phạm vi nghiên cứu, thiết lập các biến kiểm soát và các biến độc lập, xác định cấu hình phần cứng CPU Intel để tối ưu tham số (kích thước ảnh 128x128, batch size 16/32, epoch 5).
- **Phê duyệt Kiến trúc & Quyết định Kỹ thuật:** Lựa chọn ResNet18 và DenseNet121, loại bỏ VGG16 do quá tải tài nguyên CPU (138M tham số); phê duyệt chiến lược phân chia 45k train / 5k val có seed ngẫu nhiên cố định (`seed=42`).
- **Giám sát & Chỉnh sửa Mã nguồn:** Trực tiếp review diff của từng phase; yêu cầu sửa đổi thuật toán lọc tầng head trong `src/lab2/train.py` để tránh nhầm lẫn tên; bổ sung tham số `--batch-size` cho script đánh giá checkpoint.
- **Thực thi Kiểm thử & Nghiệm thu:** Tự tay chạy các lệnh kiểm thử `py -m pytest -q` (51/51 passed), kiểm tra tham số qua `scripts/inspect_models.py`, chạy smoke test và xác nhận kết quả.
- **Phần nhận xét & Đánh giá cá nhân:** [SV điền: Ghi nhận bài học kinh nghiệm, cảm nhận khi áp dụng quy trình TDD và AI-assisted software engineering trong nghiên cứu Deep Learning].

---

## 4. Các thành phần quan trọng do AI sinh ra (Important AI-Generated Components)
- `src/lab2/models.py`: Cơ chế xây dựng mô hình ResNet18 và DenseNet121, đếm tham số trainable/frozen và đóng băng/mở khóa các khối tích chập sâu theo đúng yêu cầu đề bài.
- `src/lab2/train.py`: Vòng lặp huấn luyện và validation tuân thủ nghiêm ngặt 6 bước chuẩn mực trong slide bài giảng, hỗ trợ differential learning rates và phân nhóm tham số tối ưu.
- `src/lab2/checkpoint.py`: Cơ chế lưu trữ và nạp checkpoint toàn vẹn với đầy đủ metadata, metric snapshot và cấu hình thực nghiệm.
- `src/lab2/evaluate.py`: Đánh giá tập test không thiên vị, tính per-class accuracy và vẽ ma trận nhầm lẫn seaborn heatmap.
- `scripts/check_frozen_bn.py`: Script thực nghiệm kiểm tra và chứng minh hành vi cập nhật bộ đệm động của `BatchNorm2d` khi backbone đóng băng.
- `scripts/make_results_table.py`: Script quét toàn bộ kết quả metrics JSON và tự động tổng hợp ra bảng Markdown và CSV.
- `notebooks/*.ipynb`: Toàn bộ cấu trúc tế bào code và markdown của 2 notebook bài Lab.
- `tests/`: Hệ thống 51 bài kiểm thử tự động phủ kín toàn bộ các module.

---

## 5. Mục đích của Câu lệnh Điều khiển (Prompt Purpose)
- Áp dụng nguyên tắc làm việc nghiêm ngặt: Source-First, Design-First, Test-Driven, Reproducible, CPU-First.
- Yêu cầu AI không suy đoán hoặc bịa đặt số liệu (Zero Hallucination); mọi con số phải đến từ lệnh kiểm tra thực tế trên code (`count_parameters`) hoặc kết quả thực thi thực tế.
- Kiểm soát quy trình phát triển theo từng giai đoạn (Phase) độc lập, mỗi giai đoạn kết thúc bằng một git commit nguyên tử (atomic commit) với thông điệp rõ ràng theo chuẩn Conventional Commits.

---

## 6. Chỉnh sửa và Tinh chỉnh của Sinh viên (Student Modifications)
1. **Khắc phục lỗi mã hóa ký tự trên Windows Console:** Phát hiện và hướng dẫn thêm đoạn mã cấu hình `sys.stdout.reconfigure(encoding="utf-8")` vào các script in ấn tiếng Việt để tránh lỗi `UnicodeEncodeError: 'charmap'`.
2. **Sửa đổi logic nhận diện phân tầng (Head Layer Detection):** Thay thế đoạn kiểm tra chuỗi con lỏng lẻo bằng kiểm tra tiền tố chính xác `name.startswith("fc.") or name.startswith("classifier.")` trong `src/lab2/train.py` để tránh phân loại nhầm các tầng nội bộ có tên chứa từ khóa.
3. **Chuẩn hóa đường dẫn tương đối trong CLI:** Sửa script `scripts/make_results_table.py` để tự động xác định `PROJECT_ROOT`, giúp chạy được lệnh từ bất kỳ thư mục con nào mà không bị lỗi `FileNotFoundError`.
4. **Bổ sung tham số linh hoạt cho công cụ đánh giá:** Thêm tham số `--batch-size` vào `scripts/evaluate_checkpoints.py` để có thể tùy chỉnh kích thước batch khi đánh giá trên máy có dung lượng RAM khác nhau.

---

## 7. Phương pháp Kiểm chứng (Validation)
Mọi đoạn mã do AI sinh ra hoặc chỉnh sửa đều được kiểm chứng độc lập bằng các câu lệnh thực thi trực tiếp trên hệ thống:
1. **Kiểm thử tự động:**
   ```bash
   py -m pytest -q
   # Kết quả: 51 passed in 9.85s (100% pass)
   ```
2. **Kiểm chứng tham số mô hình:**
   ```bash
   py scripts/inspect_models.py
   # Kết quả: ResNet18 Transfer (5,130 params - 0.05%), ResNet18 Fine-tune (8,398,858 params - 75.11%)
   # DenseNet121 Transfer (10,250 params - 0.15%), DenseNet121 Fine-tune (2,170,378 params - 31.17%)
   ```
3. **Kiểm chứng bộ đệm BatchNorm:**
   ```bash
   py scripts/check_frozen_bn.py
   # Kết quả: 100% buffers updated in train mode; 0% buffers updated in eval mode
   ```
4. **Kiểm tra cú pháp và thực thi Notebook:**
   ```bash
   py -m pytest tests/test_notebooks.py -v
   # Kết quả: 4/4 notebook tests passed
   ```

---

## 8. Nhật ký Đối thoại theo từng Giai đoạn (Phase Dialogue Log)

| Giai đoạn (Phase) | Mục đích Prompt (Prompt Purpose) | Đầu ra của AI (AI Output) | Chỉnh sửa của Sinh viên (Student Modification) | Phương pháp kiểm chứng (Verification) |
|---|---|---|---|---|
| **Phase 1: Baseline Audit** | Khảo sát hiện trạng repo, kiểm tra môi trường Python 3.14 và đếm tham số mô hình | Tạo `docs/progress_status.md`, chạy `pytest` ban đầu | Xác nhận các số liệu phần cứng và phiên bản thư viện | `pytest -v` (14/14 passed) |
| **Phase 2: Fixes & Missing Tests** | Sửa lỗi nhận diện tầng head trong `train.py`, thêm `--batch-size` cho `evaluate_checkpoints.py`, viết thêm tests | Cập nhật `train.py`, `evaluate_checkpoints.py`; tạo `test_config.py`, `test_evaluate.py` | Kiểm tra code diff và xác nhận logic head prefix | `pytest -v` (30/30 passed) |
| **Phase 3: History & Table Gen** | Lưu lịch sử `history.json` và tạo script tự động tổng hợp bảng kết quả markdown/csv | Thêm `save_history_json` vào `utils.py`, tạo `scripts/make_results_table.py`, viết `test_reporting.py` | Thêm xử lý `PROJECT_ROOT` và mã hóa UTF-8 cho Windows | `py scripts/make_results_table.py` và `pytest` (38/38 passed) |
| **Phase 4: Hparam Support** | Mở rộng cấu hình thêm 2 thí nghiệm siêu tham số (`resnet18_transfer_lr1e4`, `resnet18_transfer_bs32`) | Cập nhật `config.py`, `config.yaml`, `run_experiments.py`, thêm trường `group` | Tinh chỉnh CLI hỗ trợ `--group core` và `--group hparam` | `pytest -v` (46/46 passed) |
| **Phase 6: BatchNorm Audit** | Kiểm tra hành vi cập nhật bộ đệm `running_mean`/`running_var` của BatchNorm trên backbone đóng băng | Viết `scripts/check_frozen_bn.py` và bộ test `tests/test_frozen_bn.py` | Quyết định giữ nguyên code chuẩn mực theo giáo trình và tài liệu hóa hiện tượng | `py scripts/check_frozen_bn.py` và `pytest` (47/47 passed) |
| **Phase 7: Practice 2 Main Notebook** | Xây dựng notebook thực hành chính hoàn chỉnh bao quát 7 bước và 3 bài tập của đề bài | Tạo `scripts/generate_main_notebook.py`, sinh và thực thi `Lab2_Practice2_Main.ipynb` | Sửa import `sys` bị thiếu, kiểm tra đầu ra ảnh và markdown | `jupyter nbconvert --execute`, `test_notebooks.py` (49/49 passed) |
| **Phase 8: Results Analysis Notebook** | Xây dựng notebook phân tích kết quả chuyên sâu (read-only, phân tích lỗi, cặp nhầm lẫn, overfitting) | Tạo `scripts/generate_analysis_notebook.py`, sinh và thực thi `Lab2_Results_Analysis.ipynb` | Kiểm tra thuật toán trích xuất top confused class pairs | `jupyter nbconvert --execute`, `test_notebooks.py` (51/51 passed) |
| **Phase 9: Documentation Rewrite** | Viết lại toàn bộ 7 file tài liệu kỹ thuật đảm bảo không có đường dẫn cục bộ, số liệu trung thực, bảng đối chiếu | Cập nhật `README.md`, `project_description.md`, `project_design.md`, `experiment_plan.md`, `results.md`, `AI_Usage.md`, `progress_status.md` | Rà soát từng file, yêu cầu điền `*Chờ chạy full*` cho dữ liệu chưa huấn luyện | `git status`, rà soát markdown diff từng file |
