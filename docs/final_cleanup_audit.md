# FINAL CLEANUP AUDIT — DL_Lab2_Nhom1

**Ngày kiểm kê:** 2026-10-27  
**Môi trường:** Windows 11, Python 3.14.7, PyTorch 2.14.1+cpu  
**Nhánh:** `main` (đồng bộ với `origin/main`)

---

## A. Kiểm kê Repository (File Inventory)

### A1. Toàn bộ file được Git theo dõi (git ls-files)

| File | Kích thước | Mục đích | Trạng thái |
|---|---|---|---|
| `.gitignore` | 474 B | Loại trừ `__pycache__`, `*.pth`, `data/`, `runs/`, `*.tar.gz` | Giữ |
| `README.md` | 21,606 B | Tài liệu hướng dẫn toàn diện | Giữ |
| `requirements.txt` | 204 B | Danh sách phụ thuộc Python | Giữ |
| `configs/config.yaml` | 2,552 B | Cấu hình 6 thí nghiệm (4 core + 2 hparam) | Giữ |
| `docs/AI_Usage.md` | 11,128 B | Báo cáo sử dụng AI | Giữ |
| `docs/experiment_plan.md` | 7,507 B | Kế hoạch thực nghiệm | Giữ |
| `docs/progress_status.md` | 7,914 B | Báo cáo tiến độ | Giữ |
| `docs/project_description.md` | 10,574 B | Mô tả dự án và đối chiếu yêu cầu | Giữ |
| `docs/project_design.md` | 13,051 B | Tài liệu thiết kế kỹ thuật | Giữ |
| `docs/results.md` | 12,933 B | Bảng kết quả thực nghiệm | Giữ |
| `notebooks/Lab2_Practice2_Main.ipynb` | ~700 KB | Notebook thực hành chính (31 cells) | Giữ |
| `notebooks/Lab2_Results_Analysis.ipynb` | 23,606 B | Notebook phân tích kết quả | Giữ |
| `checkpoints/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `data/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `outputs/figures/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `outputs/metrics/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `runs/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `scripts/check_frozen_bn.py` | 5,697 B | Đo lường hành vi BatchNorm đóng băng (Phase 6) | Giữ |
| `scripts/evaluate_checkpoints.py` | 2,805 B | Chạy đánh giá checkpoint | Giữ |
| `scripts/generate_analysis_notebook.py` | 17,313 B | Generator cho notebook phân tích | Giữ |
| `scripts/generate_main_notebook.py` | 26,561 B | Generator cho notebook chính | Giữ |
| `scripts/inspect_models.py` | 2,478 B | Đếm và hiển thị tham số mô hình | Giữ |
| `scripts/make_results_table.py` | 8,860 B | Tổng hợp bảng kết quả từ JSON | Giữ |
| `scripts/run_experiments.py` | 6,317 B | Entry-point chạy thực nghiệm | Giữ |
| `src/lab2/*.py` (8 files) | 2–9 KB | Module mã nguồn nòng cốt | Giữ |
| `tests/*.py` (10 files) | 1–7 KB | Bộ 51 unit & integration tests | Giữ |

### A2. File Cục bộ (Không được Git theo dõi — đúng theo .gitignore)

| Đường dẫn | Kích thước | Trạng thái | Lý do |
|---|---|---|---|
| `checkpoints/*.pth` (10 files) | 29–112 MB | **Không commit — đúng** | `.gitignore` loại trừ `*.pth`; checkpoint cục bộ |
| `data/cifar-10-python.tar.gz` | 170 MB | **Không commit — đúng** | Dataset gốc; `.gitignore` loại trừ |
| `data/cifar-10-batches-py/` | ~175 MB | **Không commit — đúng** | Dữ liệu huấn luyện đã giải nén |
| `runs/**/*.tfevents.*` | 0.4–1.6 MB | **Không commit — đúng** | TensorBoard logs; `.gitignore` loại trừ |
| `__pycache__/` | Biến thiên | **Không commit — đúng** | File bytecode sinh tự động |
| `.pytest_cache/` | Nhỏ | **Không commit — đúng** | Cache pytest |

### A3. Phân loại SHA-256

**Kết quả kiểm tra:**  
Chạy `py -c "import hashlib, os; ..."` để phát hiện file nội dung trùng lặp.  
**Kết quả:** Chỉ tìm thấy 5 file `.gitkeep` trống (17 bytes, cùng hash) — đây là thiết kế có chủ đích để giữ các thư mục trống trong Git. **Không có file mã nguồn hoặc tài liệu nào trùng lặp nội dung hoàn toàn.**

---

## B. File đã Thêm Mới trong đợt Cleanup này

### B1. Hình minh họa được tạo từ dữ liệu dự án (`docs/figures/`)

| File | Kích thước | Nội dung | Nguồn dữ liệu |
|---|---|---|---|
| `docs/figures/pipeline_workflow.png` | 246 KB | Sơ đồ 6 bước quy trình thực nghiệm từ đầu đến cuối | Cấu trúc kiến trúc code thực tế của dự án |
| `docs/figures/transfer_vs_finetune.png` | 282 KB | Sơ đồ khái niệm phân biệt Transfer Learning vs Fine-Tuning: cơ chế đóng băng/mở khóa tầng và Differential LR | Sơ đồ kiến trúc khái niệm — có ghi nhận rõ không phải kết quả thực nghiệm |
| `docs/figures/model_parameters_comparison.png` | 155 KB | Biểu đồ cột Trainable vs Frozen params cho 4 cấu hình | Số liệu đo trực tiếp từ `count_parameters()` = `scripts/inspect_models.py` |

**Script tạo hình:** `scripts/generate_diagrams.py`  
**Quy tắc tuân thủ:**
- Không sử dụng số liệu giả định; tham số chính xác 100% từ `count_parameters()`.
- Sơ đồ khái niệm ghi rõ tính chất (không trình bày như kết quả thực nghiệm).
- Hình PNG 300 DPI đủ chất lượng cho báo cáo.

### B2. Cell hình ảnh được tích hợp vào `notebooks/Lab2_Practice2_Main.ipynb`

| Cell đã thêm | Vị trí | Nội dung |
|---|---|---|
| Markdown + Code (Pipeline) | Sau header/TOC (cell 1-2) | Sơ đồ quy trình toàn dự án — ghi rõ là sơ đồ kỹ thuật |
| Markdown + Code (Transfer vs FT) | Sau Step 4 Markdown (cell 11-12) | Sơ đồ khái niệm Transfer Learning vs Fine-Tuning — ghi rõ không phải kết quả thực nghiệm |
| Markdown + Code (Param Distribution) | Sau Step 4 Code (cell 13-14) | Biểu đồ tham số Trainable vs Frozen — số liệu từ source-first |

**Notebook sau khi thêm:** 31 cells (tăng từ 25 cells).  
**Notebook đã được thực thi (`nbconvert --execute --inplace`):** ✅ thành công, không có lỗi.

### B3. Script helper (không được commit vào repo chính — đề xuất)

| Script | Mục đích |
|---|---|
| `scripts/generate_diagrams.py` | Tạo 3 file PNG trong `docs/figures/` |
| `scripts/fix_notebooks.py` | Làm sạch output chứa đường dẫn máy cá nhân |
| `scripts/inject_diagrams.py` | Tiêm cell hình ảnh vào notebook |

> **Lưu ý:** `fix_notebooks.py` và `inject_diagrams.py` là script chạy một lần (one-shot). Sau khi áp dụng, chúng không cần thiết cho workflow hàng ngày. `generate_diagrams.py` hữu ích nếu muốn tái tạo hình.

---

## C. Thông tin Cá nhân đã Xử lý

### C1. Đường dẫn tuyệt đối trong notebook outputs

| Loại thông tin | File bị ảnh hưởng | Hành động |
|---|---|---|
| Đường dẫn `C:\Project\lab 2` trong output text | `notebooks/Lab2_Practice2_Main.ipynb` (cell 1 output), `notebooks/Lab2_Results_Analysis.ipynb` (cell 1 output) | Thay thế bằng `<PROJECT_ROOT>` |
| FutureWarning `torch.jit.trace` chứa đường dẫn `C:\Users\<user>\AppData\...\site-packages` | `notebooks/Lab2_Practice2_Main.ipynb` (cell output khi log TensorBoard graph) | **Dropped** output này hoàn toàn (output chỉ là cảnh báo, không chứa dữ liệu thực nghiệm) |

### C2. Username trong mã nguồn

| Thông tin | File bị ảnh hưởng | Hành động |
|---|---|---|
| `khanhlinh616/DL_Lab2_Nhom1` (tên repo) trong cell nguồn notebook | `notebooks/Lab2_Results_Analysis.ipynb` cell 14 | Đây là **tên repo GitHub công khai** cần thiết cho ngữ cảnh trích dẫn. **Giữ nguyên** — không phải thông tin cá nhân cần ẩn. |
| `khanhlinh616/DL_Lab2_Nhom1` | `scripts/generate_analysis_notebook.py` | Như trên — **giữ nguyên**. |

### C3. Thông tin trong Git history

**Kiểm tra:** `git log --pretty=format:"%h - %an <%ae> - %s"`

- Tất cả commits từ `c6034d0` trở đi dùng email `202411169+khanhlinh616@users.noreply.github.com` — đây là **email ẩn của GitHub** (GitHub private email), không tiết lộ email thực.
- Commit `4a74105 - Update README.md` dùng email `chipbong.616@gmail.com` (email cá nhân thực của người dùng, đã commit từ GitHub web UI).
- **Hành động:** Không rewrite lịch sử Git (theo quy tắc C2). Email cá nhân đã xuất hiện trong lịch sử công khai của repo; nếu cần ẩn, người dùng cần thiết lập lại Git author trên GitHub settings và bật "Block command line pushes that expose my email" trong tương lai.

### C4. Các nhóm đã kiểm tra và không tìm thấy

| Nhóm | Kết quả kiểm tra |
|---|---|
| Họ tên sinh viên cụ thể | Không tìm thấy (placeholder `[SV điền]` được giữ) |
| MSSV/mã định danh | Không tìm thấy (placeholder `[SV điền: MSSV]` được giữ) |
| Số điện thoại | Không tìm thấy |
| API key / token / password | Không tìm thấy (git grep trả về 0 kết quả) |
| Thông tin xác thực | Không tìm thấy |
| Machine name `DESKTOP-24R30PF` trong file được theo dõi | Không tìm thấy (chỉ trong `runs/*.tfevents.*` không được commit) |

---

## D. Kết quả Kiểm thử Hồi quy (Regression Tests)

| Lệnh kiểm tra | Kết quả |
|---|---|
| `py -m pytest -q` | *[Đang chạy — xem kết quả task-800]* |
| `py -m pytest tests/test_notebooks.py -v` | 9/10 passed trước khi execute notebook; 10/10 sau execute |
| `py scripts/generate_diagrams.py` | ✅ 3 PNG tạo thành công |
| `py scripts/inject_diagrams.py` | ✅ 6 cells injected, notebook 31 cells |
| `py -m jupyter nbconvert --execute --inplace notebooks/Lab2_Practice2_Main.ipynb` | ✅ exit code 0, 710,948 bytes |
| `git grep -n -i -E "api[_-]?key\|secret\|token\|password"` | ✅ 0 kết quả trong source |
| `git diff --check` | ✅ Không có whitespace errors |

---

## E. File đã Xóa

**Không có file nào bị xóa.** Kiểm tra SHA-256 và kiểm tra tham chiếu cho thấy:
- Không có file nội dung trùng lặp hoàn toàn (ngoài các `.gitkeep` trống có chủ đích).
- Mỗi script trong `scripts/` có chức năng riêng biệt và được tham chiếu trong README.md hoặc tài liệu.
- Không có file backup, temporary, hay file hệ điều hành nào được Git theo dõi.

---

## F. Trạng thái Git

- **Branch:** `main`
- **Trước cleanup:** Đồng bộ với `origin/main`
- **Sau cleanup:** Commit `chore: clean repository and improve notebook visuals` sẽ được tạo sau khi tests xác nhận.

---

## G. Các mục chưa xác minh hoặc cần quyết định của người dùng

1. **Email cá nhân trong Git history:** Commit `4a74105` có email `chipbong.616@gmail.com` trong lịch sử công khai. Để ẩn, người dùng cần: (a) Thiết lập GitHub noreply email trong Git config cục bộ, hoặc (b) Dùng `git filter-repo` + force-push — **nhưng việc này cần xác nhận rõ ràng từ người dùng và sẽ rewrite lịch sử**.
2. **Tên nhóm "Nhóm 1":** Còn trong placeholder các notebook. Người dùng cần tự điền thông tin thành viên.
3. **Script one-shot:** `scripts/fix_notebooks.py` và `scripts/inject_diagrams.py` có thể bỏ khỏi repo nếu không cần thiết cho workflow — xin quyết định từ người dùng.
