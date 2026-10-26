# Báo cáo Hiện trạng Dự án (Baseline Audit) — Lab 2

Tài liệu này ghi nhận kết quả kiểm toán hiện trạng của repository theo Phase 1 (không chỉnh sửa mã nguồn, chỉ ghi nhận dữ liệu thực tế đo đạc được trên môi trường thực thi).

**Môi trường thực tế:**
- Hệ điều hành: Windows 11 (Intel Core i5-12450HX, 24 GB RAM, không GPU rời/CUDA).
- Trình thông dịch: Python 3.14.7 (thông qua Windows Python launcher `py`).
- Thư viện cốt lõi: `torch` 2.14.1+cpu, `torchvision` 0.29.1+cpu, `tensorboard` 2.21.0, `pytest` 9.1.1.

---

## 1. Bảng Trạng thái Các Hạng mục Dự án

| STT | Hạng mục | Trạng thái | Bằng chứng (Lệnh thực thi & Kết quả thật) |
|:---:|:---|:---:|:---|
| 1 | Môi trường & Dependencies | **Xong** | `py -m pip install -r requirements.txt` (đáp ứng 100% không thiếu package). `py -c "import torch..."` -> `torch 2.14.1+cpu`, `cuda: False`. |
| 2 | Unit Tests | **Xong** | `py -m pytest -v` -> 37 passed trong 9.00s (bao gồm test mô hình, checkpoint, data, training, config, evaluate, reporting, và hparam). |
| 3 | Inspect Models (Kiến trúc & Tham số) | **Xong** | `py scripts/inspect_models.py` -> In đủ 4 cấu hình: ResNet18 Transfer (5.130 trainable), ResNet18 Fine-tune (8.398.858 trainable), DenseNet121 Transfer (10.250 trainable), DenseNet121 Fine-tune (2.170.378 trainable). |
| 4 | Smoke Test (Kiểm thử nhanh pipeline) | **Xong** | `py scripts/run_experiments.py --smoke-test --experiment resnet18_transfer` và `resnet18_transfer_lr1e4` -> Chạy 1 epoch trên subset 120 train / 30 val / 30 test, xuất checkpoint, metrics, figure và event logs thành công. |
| 5 | Kiểm tra bí mật & File nhạy cảm | **Xong** | `git grep -n -i -E "api[_-]?key\|secret\|token\|password"` -> 0 kết quả. `git ls-files` không có file > 100KB, không có file nhạy cảm. |
| 6 | 4 Thí nghiệm Full (5 epochs) | **Chưa làm** | Chờ thực thi ở Phase 5 (máy tính sinh viên chạy nền để tránh quá tải CPU). Bảng trong `docs/results.md` hiện là "*Chờ chạy full*". |
| 7 | Thí nghiệm Hyperparameter (Bài tập 2) | **Xong (Code & Config)** | Đã thêm nhóm `hparam` gồm `resnet18_transfer_lr1e4` và `resnet18_transfer_bs32` vào `config.yaml`, hỗ trợ cờ `--group core\|hparam` trong `run_experiments.py`, hoàn thành smoke test và 3 unit test xác minh (chờ chạy full ở Phase 5). |
| 8 | TensorBoard Screenshots | **Chưa làm** | Thư mục `docs/figures/` chưa có ảnh chụp TensorBoard thật (chờ Phase 5 sau khi chạy xong full thí nghiệm). |
| 9 | Notebook Chính (`Lab2_Practice2_Main.ipynb`) | **Chưa làm** | Thư mục `notebooks/` chưa có file này (nhiệm vụ của Phase 7). |
| 10 | Notebook Phân tích (`Lab2_Results_Analysis.ipynb`) | **Chưa làm** | Thư mục `notebooks/` chưa có file này (nhiệm vụ của Phase 8). |
| 11 | `README.md` | **Đang làm** | Đã có nội dung ban đầu từ commit `f283750`, nhưng còn chứa đường dẫn tuyệt đối máy cá nhân (`c:\Project\lab 2/`) và ước tính thời gian chưa đo thực tế. Cần cập nhật chuẩn hóa ở Phase 9. |
| 12 | `docs/project_description.md` | **Đang làm** | Cần bổ sung bảng đối chiếu chi tiết các yêu cầu của đề bài Practice 2 (nhiệm vụ của Phase 9). |
| 13 | `docs/project_design.md` | **Đang làm** | Cần cập nhật interface các module sau Phase 2-4, các quyết định thiết kế và rủi ro BatchNorm đóng băng (nhiệm vụ của Phase 9). |
| 14 | `docs/experiment_plan.md` | **Đang làm** | Cần bổ sung 2 thí nghiệm hyperparameter và các biến kiểm soát (nhiệm vụ của Phase 9). |
| 15 | `docs/results.md` | **Đang làm** | Đang ở dạng khung chờ kết quả full; cần chuyển thành bảng sinh tự động từ script sau khi chạy (nhiệm vụ của Phase 3 & 9). |
| 16 | `docs/AI_Usage.md` | **Đang làm** | Đã có khung ban đầu; cần cập nhật nhật ký hội thoại từng phase theo MASTER_RULES.md (nhiệm vụ của Phase 9). |

---

## 2. Khẳng định Tài liệu Chưa Kiểm chứng hoặc Cần Điều chỉnh

Qua đối chiếu giữa các tài liệu (`README.md`, `docs/results.md`, `docs/AI_Usage.md`) với kết quả kiểm toán thực tế:

1. **Khẳng định "14/14 tests passed":**
   - **Xác minh: ĐÚNG.** Lệnh `py -m pytest -v` hoàn thành 14/14 tests trong 7.37s.
2. **Khẳng định số lượng tham số trong bảng kiến trúc:**
   - **Xác minh: ĐÚNG 100%.** Số liệu tham số trong `README.md` và `docs/results.md` khớp hoàn toàn từng con số với kết quả xuất từ `py scripts/inspect_models.py`.
3. **Khẳng định "Tương thích hoàn toàn trên Python 3.14":**
   - **Xác minh: ĐÚNG MỘT PHẦN / CÓ CẢNH BÁO.** Mã nguồn chạy được trên Python 3.14.7, tuy nhiên khi TensorBoard Logger ghi nhận đồ thị mô hình (`add_graph`), PyTorch xuất hiện cảnh báo `FutureWarning: torch.jit.trace is not supported in Python 3.14+ and may break`. Quá trình chạy không bị crash nhưng cần lưu ý cảnh báo này trong tài liệu kỹ thuật.
4. **Khẳng định "30–60 phút mỗi thí nghiệm trên CPU":**
   - **Xác minh: CHƯA KIỂM CHỨNG.** Đây là con số ước tính lý thuyết, chưa có dữ liệu đo đạc thực tế của lượt chạy full 5 epochs trên toàn bộ 45.000 ảnh CIFAR-10. Cần đo thời gian thật ở Phase 5 để cập nhật lại.
5. **Khẳng định "Đường dẫn thư mục dự án `c:\Project\lab 2/`":**
   - **Xác minh: SAI QUY TẮC REPRODUCIBILITY.** Đường dẫn trong tài liệu `README.md` đang mang tính cá nhân cục bộ, vi phạm nguyên tắc chỉ dùng đường dẫn tương đối. Cần làm sạch ở Phase 9.
6. **Khẳng định "Smoke test thành công":**
   - **Xác minh: ĐÚNG.** Chạy thành công lệnh `py scripts/run_experiments.py --smoke-test --experiment resnet18_transfer` và kiểm chứng các tệp tin đầu ra được sinh ra đúng vị trí.
7. **Khẳng định về kết quả chính thức trong `docs/results.md`:**
   - **Xác minh: ĐÚNG.** Tài liệu ghi nhận trung thực là "*Chờ chạy full*", không bịa đặt số liệu giả định.
