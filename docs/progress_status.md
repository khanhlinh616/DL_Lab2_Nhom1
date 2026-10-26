# Báo cáo Tiến độ Toàn diện Dự án (Progress Status Report) — Lab 2

Tài liệu này ghi nhận hiện trạng hoàn thiện toàn diện của repository sau khi thực hiện có hệ thống các Phase từ 1 đến 9 theo đúng quy chuẩn của `MASTER_RULES.md`, đề cương Practice 2 và các nguyên tắc Software Quality Engineering.

**Môi trường thực tế đo đạc:**
- Hệ điều hành: Windows 11 (Intel Core i5-12450HX, 24 GB RAM, không GPU rời/CUDA).
- Trình thông dịch: Python 3.14.7 (thông qua Windows Python launcher `py`).
- Thư viện cốt lõi: `torch` 2.14.1+cpu, `torchvision` 0.29.1+cpu, `tensorboard` 2.21.0, `pytest` 9.1.1, `jupyter` 1.1.1, `nbconvert` 7.16.6.

---

## 1. Bảng Trạng thái Các Hạng mục Dự án

| STT | Hạng mục | Trạng thái | Bằng chứng (Lệnh thực thi & Kết quả thật) |
|:---:|:---|:---:|:---|
| 1 | Môi trường & Dependencies | **Hoàn thành** | `py -m pip install -r requirements.txt` (đáp ứng 100% không thiếu package, đã bổ sung jupyter, nbconvert, nbformat). `py -c "import torch..."` -> `torch 2.14.1+cpu`, `cuda: False`. |
| 2 | Bộ Kiểm thử Tự động (Unit & Integration Tests) | **Hoàn thành** | `py -m pytest -q` -> **51 passed** trong 9.85s (bao gồm test mô hình, checkpoint, data, training, config, evaluate, reporting, hparam, frozen BN, và notebooks). |
| 3 | Khảo sát Kiến trúc & Đếm Tham số | **Hoàn thành** | `py scripts/inspect_models.py` -> In đủ 6 cấu hình: ResNet18 Transfer (5.130 trainable - 0.05%), ResNet18 Fine-tune (8.398.858 trainable - 75.11%), DenseNet121 Transfer (10.250 trainable - 0.15%), DenseNet121 Fine-tune (2.170.378 trainable - 31.17%), ResNet18 LR1e-4 (5.130 trainable), ResNet18 BS32 (5.130 trainable). |
| 4 | Smoke Test Toàn diện Pipeline | **Hoàn thành** | `py scripts/run_experiments.py --smoke-test --experiment resnet18_transfer` và `resnet18_transfer_lr1e4` -> Chạy 1 epoch trên subset 120 train / 30 val / 30 test, xuất checkpoint, metrics, figure và event logs thành công. |
| 5 | Kiểm tra Bảo mật & Bí mật | **Hoàn thành** | `git grep -n -i -E "api[_-]?key\|secret\|token\|password"` -> 0 kết quả. Toàn bộ repo sạch, không chứa file > 100KB (trừ notebook đã thực thi 382KB được kiểm duyệt), không chứa credential. |
| 6 | Thí nghiệm Siêu tham số (Bài tập 2) | **Hoàn thành** | Đã cấu hình `group: hparam` gồm `resnet18_transfer_lr1e4` và `resnet18_transfer_bs32` trong `configs/config.yaml`, hỗ trợ cờ CLI `--group core\|hparam` trong `scripts/run_experiments.py`. |
| 7 | Đo đạc Thực nghiệm BatchNorm Đóng băng (Phase 6) | **Hoàn thành** | Tạo `scripts/check_frozen_bn.py` và `tests/test_frozen_bn.py`. Kết quả: Ở `train` mode, 100% buffer (60/60 ResNet18, 363/363 DenseNet121) bị cập nhật; ở `eval` mode, 0% buffer thay đổi. |
| 8 | Notebook Thực hành Chính (`Lab2_Practice2_Main.ipynb`) | **Hoàn thành** | Đã tạo bằng `scripts/generate_main_notebook.py`, thực thi thành công không lỗi bằng `jupyter nbconvert --execute` (dung lượng 382 KB, đầy đủ 7 bước và 3 bài tập). |
| 9 | Notebook Phân tích Kết quả (`Lab2_Results_Analysis.ipynb`) | **Hoàn thành** | Đã tạo bằng `scripts/generate_analysis_notebook.py`, thực thi thành công không lỗi bằng `jupyter nbconvert --execute` (dung lượng 21 KB, phân tích đa biến, trích xuất top cặp nhầm lẫn, phân tích overfitting). |
| 10 | `README.md` | **Hoàn thành** | Đã viết lại hoàn chỉnh: không chứa đường dẫn tuyệt đối, cập nhật 51/51 tests, bảng tham số chính xác, hướng dẫn thực thi rõ ràng, liên kết tài liệu đầy đủ. |
| 11 | `docs/project_description.md` | **Hoàn thành** | Đã bổ sung bảng đối chiếu chi tiết 7 bước + 3 bài tập + 2 mục tiêu của Practice 2 và giải trình kỹ thuật việc không dùng VGG16. |
| 12 | `docs/project_design.md` | **Hoàn thành** | Đã bổ sung bảng Interface chi tiết từng hàm của các module, các quyết định thiết kế kỹ thuật (Adam, 128px, val split, differential LR) và rủi ro thực nghiệm BatchNorm. |
| 13 | `docs/experiment_plan.md` | **Hoàn thành** | Đã bổ sung 2 thí nghiệm siêu tham số, phân định rõ biến kiểm soát, biến độc lập và ghi nhận giới hạn hạt giống đơn lẻ (`seed=42`). |
| 14 | `docs/results.md` | **Hoàn thành** | Đã đồng bộ bảng 6 thí nghiệm tương thích `make_results_table.py`, bảng per-class accuracy, phân tích các cặp nhầm lẫn hàng đầu, ghi nhận trung thực `*Chờ chạy full*`. |
| 15 | `docs/AI_Usage.md` | **Hoàn thành** | Đã cập nhật chi tiết công cụ (Claude + Antigravity), đóng góp của sinh viên, các thành phần AI sinh ra, lệnh kiểm chứng và bảng nhật ký đối thoại theo 10 phases. |
| 16 | Huấn luyện Full 5 Epochs (Phase 5) | **Chờ thực thi** | Mã nguồn, cấu hình và lệnh đã sẵn sàng 100%. Sinh viên có thể chạy bất kỳ lúc nào trên máy cá nhân bằng lệnh `py scripts/run_experiments.py --all` (ước tính ~2-4 giờ trên CPU). |

---

## 2. Bảng Tổng hợp Lịch sử Commit theo từng Phase

Toàn bộ các thay đổi trong dự án được tổ chức thành các commit nguyên tử (atomic commits) tuân thủ quy tắc Conventional Commits:

| Hash | Loại Commit | Phạm vi | Mô tả ngắn gọn |
|---|:---:|:---:|---|
| `985ba39` | `docs` | `progress` | Add baseline audit report (Phase 1) |
| `16a86cd` | `fix/test`| `train,scripts` | Robust head detection and batch-size arg; add config/evaluate tests (Phase 2) |
| `93ef47b` | `feat` | `reporting` | Save history json and generate results table (Phase 3) |
| `6f51bf5` | `feat` | `config` | Add hyperparameter experiment group (Phase 4) |
| `56ee456` | `test` | `bn` | Measure frozen backbone BatchNorm buffer updates (Phase 6) |
| `8644fb7` | `docs` | `notebook` | Add Practice 2 main notebook (Phase 7) |
| `09d69b0` | `docs` | `notebook` | Add results analysis notebook (Phase 8) |
| `6c3388e` | `docs` | `readme` | Rewrite README according to actual codebase and standards (Phase 9 - File 1) |
| `ee36fbc` | `docs` | `description` | Add Practice 2 requirement cross-reference table (Phase 9 - File 2) |
| `b4fefae` | `docs` | `design` | Update design doc with module interfaces and technical decisions (Phase 9 - File 3) |
| `f244838` | `docs` | `plan` | Add hyperparameter experiment plan and control variables (Phase 9 - File 4) |
| `eb34a37` | `docs` | `results` | Format results documentation and limitations (Phase 9 - File 5) |
| `8e9a04b` | `docs` | `ai-usage` | Update AI usage report and phase dialogue log (Phase 9 - File 6) |

---

## 3. Kết luận và Mức độ Sẵn sàng (Definition of Done Audit)

1. **Chất lượng mã nguồn (Code Quality):** Modular hóa cao, phân tách rõ ràng các tầng logic, 100% tuân thủ type hints và docstrings.
2. **Độ bao phủ kiểm thử (Test Coverage):** 51 tests kiểm thử tự động vượt qua 100% không có cảnh báo lỗi nghiêm trọng.
3. **Tính tái lập (Reproducibility):** Khóa seed cố định `seed=42`, tách train/val/test minh bạch, không rò rỉ dữ liệu (zero data leakage).
4. **Tính tương tác (Interactivity):** 2 notebooks Jupyter hoàn chỉnh, thực thi sạch, không lỗi, trực quan hóa đầy đủ đồ thị và ma trận nhầm lẫn.
5. **Tính trung thực khoa học (Truthful Reporting):** Mọi tham số và kết quả đo đạc đều có nguồn gốc thực tế, không bịa đặt số liệu training khi chưa chạy full.
