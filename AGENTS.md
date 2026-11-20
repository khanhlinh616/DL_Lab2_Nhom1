# MASTER RULES & PROJECT WORKFLOW GUIDELINES

## PHẦN I: WORKFLOW & DEFINITION OF DONE

### 1. Definition of Done (DoD)
Task chỉ hoàn thành khi đáp ứng đầy đủ:
- **Đủ yêu cầu:** Đạt đầy đủ các yêu cầu chức năng.
- **Code chạy:** Thực thi không lỗi cú pháp hay runtime.
- **Test passed:** Toàn bộ kiểm thử đều vượt qua (100% passed).
- **No secrets:** Không để lộ API key, token, mật khẩu trong code/commit.
- **Code review:** Mã nguồn được tự rà soát, đạt chuẩn chất lượng.
- **GitHub:** Trạng thái mã nguồn đồng bộ với GitHub.
- **Docs updated:** Tài liệu được cập nhật tương ứng thay đổi.
- **Committed:** Đã commit theo chuẩn Conventional Commits.
> **Lưu ý:** Viết code xong phải kiểm tra, đánh giá kỹ lưỡng thay vì chỉ thấy chạy được là xong. Cập nhật tiến độ vào file markdown (`.md`).

### 2. Các mục bắt buộc trong báo cáo: Example In-out | Results | Limitations | Future Improvements.
### 3. AI Usage: Ghi nhận Tools used, Main purposes, Student contributions, Important AI components, Prompt purpose, Student modification, Validation, Context đối chiếu.
### 4. Lưu ý khi dùng AI: Không để AI đẩy code khi chưa chạy thử/chưa hiểu; tạo test suite passed thực chất; không lộ API key/token; không commit .env; không sửa số liệu thực nghiệm.
### 5. Prompt tốt: `Context nhỏ + Mục tiêu rõ ràng + Tiêu chí chung cụ thể`.
### 6. Plan: `Mục tiêu + In/Out + Feature` &rarr; `Constraints → Architecture + Modules + Milestones + Risks → Chia nhỏ viết code`.
### 7. Implement: Xác định rõ `Module + Code + In/Out + Interface + Constraints`.
### 8. Sử dụng và Tài liệu:
- **README.md:** Overview, Problem, Features, Architecture, Folder structure, Install/Run/Test, Model, Results, Limitations, Improvements, AI Usage.
- **Tài liệu trong `docs/`:** `labs2.md` (12 phần), `data.md` (10 phần), `model.md` (đặc tả 100% info), `training.md`, `evaluation.md`, `audit-log.md` (5 giai đoạn), `plan-progress-status.md`.
### 9. Build MVP: MVP trước; một module/một lần; một thay đổi/một lần. Quy trình: `Describe → Generate → Review → Run (Smoke test) → Dry-run → Test → Refactor → Commit`.

---

## PHẦN II: MASTER RULES — QUY TẮC CHUNG CHO MỌI DỰ ÁN

### 1. Nguyên tắc làm việc
- Đọc kỹ yêu cầu, rubric và mã nguồn trước khi làm; xác định rõ mục tiêu, phạm vi, I/O, ràng buộc kỹ thuật.
- Ưu tiên tài liệu chính thức; không tự ý thêm yêu cầu hay mở rộng phạm vi; không bịa thông tin, số liệu, tính năng.
- Phân biệt rõ thông tin tài liệu, kiểm chứng, giả định và đề xuất; kiểm tra nguồn có sẵn trước khi hỏi.

### 2. Đọc và tuân thủ tài liệu nguồn
- Đọc đúng nội dung nguồn; giữ nguyên thuật ngữ chuyên môn, công thức, ký hiệu, cấu trúc lập luận.
- Không âm thầm thay thế nội dung nguồn bằng kiến thức ngoài; nếu bổ sung phải phân biệt rõ ràng.
- Đối chiếu từng tiêu chí rubric trước khi kết luận hoàn thành; không khẳng định nếu chưa đủ căn cứ.

### 3. Quy tắc chỉnh sửa dự án có sẵn
- Kiểm tra cấu trúc, cấu hình, mã nguồn, tài liệu, test trước khi sửa; xác định đúng vấn đề cần sửa.
- Ưu tiên chỉnh sửa tối thiểu, có mục tiêu, kiểm chứng được; giữ nguyên phần đang hoạt động đúng.
- Không đổi tên file/hàm/biến/API/I/O nếu không cần thiết; không xóa dữ liệu, checkpoint, cấu hình có giá trị.
- Đánh giá ảnh hưởng và phụ thuộc trước khi sửa thành phần dùng chung; cấu trúc dựa trên thực tế.

### 4. Quy tắc chất lượng mã nguồn
- Code rõ ràng, nhất quán, dễ đọc, dễ bảo trì và kiểm thử; phân tách trách nhiệm hợp lý (SRP).
- Tên biến/hàm/class thể hiện đúng mục đích; xử lý input không hợp lệ và trường hợp biên phù hợp.
- Không thêm abstraction/thư viện thừa; không hardcode secrets; tôn trọng phiên bản ngôn ngữ và môi trường.

### 5. Quy tắc dữ liệu và kết quả
- Xác định rõ nguồn gốc, kiểu dữ liệu, ý nghĩa trước khi dùng; không tự tạo hay làm đẹp số liệu.
- Kiểm tra missing, duplicates, outliers, kiểu dữ liệu khi có ý nghĩa; giữ lại dữ liệu gốc để kiểm tra.
- Phân biệt train/val/test, chống rò rỉ dữ liệu (data leakage); mọi bảng/biểu đồ phải dựa trên dữ liệu thật.

### 6. Quy tắc lựa chọn giải pháp kỹ thuật
- Chọn giải pháp dựa trên bài toán, độ phức tạp, tài nguyên và khả năng kiểm chứng; giải thích lý do khi cần.
- Không mặc định một thuật toán/framework là tốt nhất; chỉ tối ưu hóa khi có nhu cầu hoặc bằng chứng thực tế.
- Cân nhắc độ chính xác, bảo trì, hiệu năng; không tự ý đổi phương pháp đề bài đã bắt buộc.

### 7. Kiểm thử và xác minh
- Chạy test hiện có trước khi sửa để biết baseline; sau khi sửa, chạy test liên quan trực tiếp.
- Kiểm tra cả input hợp lệ, trường hợp biên và lỗi; kiểm tra tương thích giữa các thành phần.
- Không chỉ dựa vào cú pháp không lỗi; không bỏ qua warning/lỗi có ý nghĩa; báo cáo chính xác test đã chạy.

### 8. Quy tắc tài liệu và văn phong
- Tài liệu phục vụ người đọc hiểu cơ chế và kết quả; câu trực tiếp, tự nhiên, cô đọng, giàu thuật ngữ kỹ thuật.
- Loại bỏ câu sáo rỗng, tự khen, lặp ý; không tự nhận xét mô hình tốt/tối ưu nếu chưa có bằng chứng.
- Giữ thuật ngữ nhất quán; không thêm phần giới thiệu/kết luận thừa nếu đề không yêu cầu.

### 9. Quy tắc tổ chức dự án
- Tổ chức file theo chức năng thực tế; phân biệt source, data, config, test, docs, artifacts.
- Tránh cấu trúc quá sâu hoặc quá nhiều tầng; không giữ file tạm, cache, bản sao trùng lặp trong bàn giao.

### 10. Tài nguyên và môi trường thực thi
- Kiểm tra giới hạn phần cứng, RAM, storage, thời gian chạy; ưu tiên giải pháp chạy được trên máy người dùng.
- Không mặc định có GPU/Internet/admin; hạn chế tải/tính toán lại; chỉ thêm dependency thực sự cần.

### 11. Bảo mật và độ tin cậy
- Không để lộ thông tin xác thực, secrets trong mã nguồn và docs; không thực hiện thao tác phá hủy dữ liệu.
- Không vô hiệu hóa kiểm tra/xử lý lỗi chỉ để pass test; không sửa output để giả mạo hệ thống đúng.

### 12. Quy trình thực hiện công việc
`Phân tích yêu cầu → Đọc tài liệu/workspace → Xác định trạng thái & ràng buộc → Xác định phần cần làm → Chọn phương án tối thiểu → Thực hiện trực tiếp → Kiểm tra tương thích → Chạy test → Rà soát → Báo cáo kết quả`.

### 13. Quy tắc báo cáo kết quả
Báo cáo ngắn gọn, đủ thông tin: Việc đã làm, file đã đổi, test thực tế đã chạy và kết quả, giới hạn còn tồn tại, quyết định cần xác nhận. Không bịa việc chưa làm thành đã làm; không viết dài dòng.

### 14. Quy tắc ưu tiên khi có xung đột
1. An toàn, bảo mật & toàn vẹn dữ liệu &rarr; 2. Yêu cầu bắt buộc của người dùng/đề bài &rarr; 3. Tính chính xác kiến thức/số liệu &rarr; 4. Khả năng chạy đúng & tương thích &rarr; 5. Kiểm thử & bảo trì &rarr; 6. Tính đơn giản & hiệu quả &rarr; 7. Sở thích trình bày. Nêu rõ vấn đề nếu có xung đột không thể tự giải quyết an toàn.

### 15. Chỉ dẫn cốt lõi cuối cùng
Ưu tiên tính chính xác, khả năng kiểm chứng, thay đổi tối thiểu và kết quả thực tế. Không bịa, không mở rộng phạm vi tùy tiện, không phá vỡ phần đang hoạt động tốt. Thực hiện trực tiếp khi đủ thông tin; chỉ hỏi khi cần quyết định quan trọng.

---

## PHẦN III: MASTER RULES CHO BÁO CÁO HỌC THUẬT & TÀI LIỆU KHOA HỌC

### 1. Master Rule — Deep Learning Scientific Report (`docs/labs2.md`)
- **Chuẩn học thuật:** Phong cách bài báo khoa học, công thức toán học chuẩn, câu từ cô đọng, thuật ngữ kỹ thuật chính xác.
- **12 phần bắt buộc:** (1) Abstract; (2) Introduction (Bối cảnh, Mục tiêu, Vấn đề); (3) Related Work; (4) Theoretical Background (Residual Shortcuts, Dense Concatenation, Loss, Adam, Differential LR); (5) Methodology (CIFAR-10, Preprocessing 128, Layer breakdown); (6) Experimental Setup (CPU-First, Hyperparameters, Metrics); (7) Results & Evaluation (Bảng 6 cấu hình, liên kết 8 biểu đồ); (8) Comparative Analysis (Transfer vs Fine-tuning, ResNet vs DenseNet, LR, Batch Size); (9) Discussion & Limitations (Ma trận nhầm lẫn cat/dog, automobile/truck; giới hạn CPU); (10) Conclusion & Future Work; (11) References; (12) Appendix (Output Rule & CLI commands).
- **Output Rule:** Tách biệt Notebook và Báo cáo độc lập; kết quả lưu tại `outputs/figures/`, `outputs/metrics/`, `outputs/results_table.md`, `outputs/Lab2_Report.html`.

### 2. Master Rule — Model Documentation (`docs/model.md`)
- **Nguyên tắc:** *"Preserve 100% Technical Information – Improve Presentation Only"*.
- **Bảo toàn đầy đủ:** Purpose, Pretrained source (`IMAGENET1K_V1`), Kaiming Uniform, Freeze/Unfreeze, Differential LR ($\eta_{\text{backbone}}=0.1\times\eta_{\text{head}}$), Architecture & Stages, Bảng Parameter Counts 4 cấu hình cốt lõi, BatchNorm behavior (`train` mode buffer cập nhật vs `eval` mode buffer tĩnh), I/O shapes, Source Files mapping, Verification commands.

### 3. Master Rule — Scientific Dataset Presentation (`docs/data.md`)
- **10 phần chuẩn khoa học:** (1) Tổng quan Dataset; (2) Nguồn gốc (Toronto, Krizhevsky/Hinton 2009); (3) Cấu trúc dữ liệu & Class Mapping; (4) Đánh giá chất lượng 6 tiêu chí (0% lỗi); (5) Tiền xử lý (Split 45k/5k/10k, Resize 128, Augmentation, ToTensor, Normalize); (6) EDA (Bảng thống kê R/G/B, Phân phối, Tương quan $r>0.88$); (7) Trực quan hóa 7 dạng biểu đồ (Bar, Hist, Box, Heatmap, Scatter, Line Scree, PCA 2D Scatter chứng minh không thể phân tách tuyến tính trên pixel thô); (8) 5 Phát hiện then chốt; (9) Hạn chế dataset; (10) Kết luận.
