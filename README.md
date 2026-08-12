# DRMD trên LAMDA

Mã nguồn thí nghiệm DRMD (Dynamic Reject for Malware Detection) trên bộ dữ liệu [LAMDA](https://huggingface.co/datasets/IQSeC-Lab/LAMDA), kèm notebook huấn luyện chạy được trên **Kaggle** và **Google Colab**.

## Nội dung kho

| Đường dẫn | Mô tả |
|---|---|
| `notebook_drmd_lamda_unified_v14.ipynb` | Notebook huấn luyện thống nhất (một tệp duy nhất) |
| `references/DRMD` | Mã DRMD gốc (khi đóng gói thành Kaggle Dataset) |
| `references/tesseract-ml-release` | Thư viện temporal split |
| `references/RPAL` | Phụ thuộc liên quan |

Khi đẩy lên Kaggle Dataset `thanhngo1007/drmd-lamda-dataset`, cấu trúc thư mục gốc cần có `references/DRMD` để notebook tự tìm project.

## Năm phương pháp trong notebook

| Phương pháp | Mô tả ngắn |
|---|---|
| Static-MLP | MLP cố định trên cửa sổ huấn luyện ban đầu |
| DRMD-IRAL | Phản hồi nội sinh \(Q_t = R_t\) |
| DRMD-IRAAL | Cập nhật có ngân sách B = 15 |
| DRMD-FN | IRAAL + phạt false-negative thích nghi |
| DRMD-FN-BHR | DRMD-FN + Balanced Hard Replay |

Mỗi phương pháp chạy trên 10 seed cố định → **50 lượt**. Báo cáo chỉ hợp lệ khi đủ 50/50 artifact.

Tham số cố định: B = 15, chi phí từ chối −0,1, mp = 1, bộ nhớ fine-tune 5.000 mẫu, 110 chu kỳ kiểm thử, 885.947 mẫu test theo revision LAMDA đã khóa.

Ba chỉnh sửa so với bản FN cũ:

1. Phạt FN chỉ áp dụng cho hành động phân loại trực tiếp; phần thưởng khi từ chối giữ như DRMD gốc.
2. Bộ điều khiển FN dùng hậu nghiệm Beta, chỉ đổi mức phạt khi khoảng tin cậy đủ hỗ trợ.
3. Ở FN-BHR, tỷ lệ phát lại mẫu FN giảm theo căn bậc hai của hệ số phạt.

---

## Input bắt buộc

### 1. Dataset mã nguồn và cấu hình

- Kaggle: [thanhngo1007/drmd-lamda-dataset](https://www.kaggle.com/datasets/thanhngo1007/drmd-lamda-dataset)
- Bên trong phải có thư mục `DRMD_LAMDA_DATASET/references/DRMD` (hoặc tương đương sau khi mount).

### 2. Dataset đặc trưng LAMDA đã xử lý

- Kaggle: [thanhngo1007/lamda-full-processed](https://www.kaggle.com/datasets/thanhngo1007/lamda-full-processed)
- Bên trong có thư mục `Baseline` với các file parquet theo năm (train/test), khớp SHA-256 trong notebook.

### 3. Checkpoint (tùy chọn)

Nếu đã có zip checkpoint từ phiên trước (`*_checkpoint_raw.zip`), gắn thêm dataset chứa zip đó để notebook khôi phục và bỏ qua các lượt đã hoàn tất.

---

## Chạy trên Kaggle

1. Tạo notebook mới (hoặc upload `notebook_drmd_lamda_unified_v14.ipynb`).
2. **Settings**
   - Accelerator: **GPU** (bắt buộc; notebook kiểm tra CUDA).
   - Internet: tắt cũng được nếu đã gắn đủ dataset; bật nếu cần cài package lần đầu.
3. **Add Input**
   - `thanhngo1007/drmd-lamda-dataset`
   - `thanhngo1007/lamda-full-processed`
   - (Tùy chọn) dataset checkpoint.
4. Chạy lần lượt từng ô, hoặc **Run All**.
5. Kết quả nằm dưới `/kaggle/working/results/...`:
   - `tables/` — CSV tổng hợp, so sánh ghép cặp
   - `figures/` — hình (khi đủ dữ liệu)
   - `*_checkpoint_raw.zip` — checkpoint để gắn phiên sau
   - `*_full_test_report_bundle.zip` — gói báo cáo khi đủ 50/50

Thời gian tham khảo: khoảng 4–6 phút mỗi lượt DRMD trên GPU Kaggle; 50 lượt có thể cần nhiều phiên. Notebook tự cắt theo `SESSION_LAUNCH_CUTOFF_HOURS` (mặc định 10,5 giờ) và xuất checkpoint cuối ô huấn luyện.

---

## Chạy trên Google Colab

1. Upload notebook lên Colab.
2. Runtime → Change runtime type → **GPU**.
3. Tạo secret (biểu tượng chìa khóa):
   - Tên: `KAGGLE_API_TOKEN`
   - Giá trị: token API Kaggle (không dán token vào ô code).
4. (Tùy chọn) Mount Google Drive nếu muốn sao lưu zip tự động.
5. Chạy **Ô 1**. Notebook sẽ:
   - Cài `kagglehub` nếu thiếu
   - Tải hai dataset bắt buộc vào `/content/drmd_colab/input`
   - Dùng `/content/drmd_colab/working` làm thư mục ghi
6. Chạy tiếp các ô còn lại theo thứ tự.

Checkpoint tùy chọn:

- Đặt `COLAB_CHECKPOINT_DATASET = "owner/slug"` trong Ô 1, hoặc
- Tạo secret `KAGGLE_CHECKPOINT_DATASET` với cùng giá trị.

Sao lưu Drive (nếu đã mount): zip được copy vào  
`/content/drive/MyDrive/DRMD_checkpoints/` với tên có pha và thời điểm UTC, không ghi đè bản cũ.

---

## Thứ tự ô trong notebook

| Ô | Việc làm |
|---|---|
| 1 | Cấu hình giao thức, seed, ma trận phương pháp; bootstrap Colab |
| 2 | Khôi phục checkpoint nếu có |
| 3 | Kiểm kê artifact đã khôi phục |
| 4 | Ghi `UndefinedMetricWarning` ra file thay vì in tràn log |
| 5 | Cố định random / NumPy / PyTorch / cuDNN |
| 6 | Cài dependency DRMD và vá runtime |
| 7 | Nạp LAMDA, kiểm SHA-256 và split |
| 8 | Huấn luyện Static-MLP (10 seed) |
| 9 | Huấn luyện IRAL, IRAAL, FN, FN-BHR (40 seed-lượt) |
| 10 | Tổng hợp monthly, summary, so sánh ghép cặp |
| 11 | Điều kiện đọc kết quả |
| 12 | Đóng gói zip bảng / checkpoint |

Không đảo thứ tự ô: ô sau phụ thuộc biến và artifact của ô trước.

---

## Đọc kết quả

Sau Ô 10, xem:

- `unified_integrity_summary.json` — `report_ready`, số lượt hợp lệ, lỗi
- `unified_all_methods_overview.csv` — trung bình ± sd theo phương pháp
- `unified_paired_effects_summary.csv` — hiệu ghép cặp (FN − IRAAL, FN-BHR − FN, …)

Chỉ diễn giải hiệu năng khi `report_ready` là true (đủ 50/50).

---

## Pha chạy và tiếp tục phiên

Trong Ô 1:

```python
KAGGLE_RUN_PHASE = "phase_a_primary"   # huấn luyện
# KAGGLE_RUN_PHASE = "aggregate_only"  # chỉ tổng hợp từ artifact đã có
```

Gắn lại checkpoint zip từ phiên trước để bỏ qua lượt đã xong. Không trộn artifact từ giao thức khác (chữ ký và `protocol_version` được kiểm tra).

---

## Lưu ý

- GPU bắt buộc khi huấn luyện (`REQUIRE_CUDA = True`).
- Seed cố định: `[0, 1, 7, 13, 26, 42, 73, 2026, 314159, 281083886]`.
- Không sửa `EXPECTED_LAMDA_FILE_SHA256` hay revision HF trừ khi chủ đích đổi bộ dữ liệu.
- Notebook tự nhúng mã BHR và các vá runtime; không cần package `experiments` có sẵn trên Dataset.

## Liên kết

- Dataset đặc trưng: [LAMDA trên Hugging Face](https://huggingface.co/datasets/IQSeC-Lab/LAMDA)
- Kaggle dataset mã nguồn: [thanhngo1007/drmd-lamda-dataset](https://www.kaggle.com/datasets/thanhngo1007/drmd-lamda-dataset)
- Kaggle dataset parquet: [thanhngo1007/lamda-full-processed](https://www.kaggle.com/datasets/thanhngo1007/lamda-full-processed)
