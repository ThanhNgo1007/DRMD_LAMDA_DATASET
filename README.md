# DRMD trên LAMDA — Hướng dẫn chạy

Notebook huấn luyện thống nhất cho 5 phương pháp trên bộ dữ liệu LAMDA.  
Chạy được trên **Kaggle** và **Google Colab**.

| Phương pháp | Mô tả |
|---|---|
| Static-MLP | MLP cố định trên cửa sổ huấn luyện ban đầu |
| DRMD-IRAL | Phản hồi nội sinh \(Q_t = R_t\) |
| DRMD-IRAAL | Cập nhật theo ngân sách B = 15 |
| DRMD-FN | IRAAL + phạt false-negative thích nghi |
| DRMD-FN-BHR | DRMD-FN + Balanced Hard Replay |

Mỗi phương pháp chạy 10 seed → **50 lượt**. Báo cáo chỉ hợp lệ khi đủ 50/50.

---

## 1. Chuẩn bị dataset (bắt buộc)

Cần **2 dataset Kaggle**:

| Dataset | Link | Vai trò |
|---|---|---|
| `thanhngo1007/drmd-lamda-dataset` | [Link](https://www.kaggle.com/datasets/thanhngo1007/drmd-lamda-dataset) | Mã nguồn DRMD + cấu hình |
| `thanhngo1007/lamda-full-processed` | [Link](https://www.kaggle.com/datasets/thanhngo1007/lamda-full-processed) | File parquet đặc trưng (train/test theo năm) |

**Checkpoint (tùy chọn):**  
Nếu đã có file `*_checkpoint_raw.zip` từ lần chạy trước, tạo dataset riêng chứa file đó để notebook tự khôi phục và bỏ qua các lượt đã xong.

---

## 2. Chạy trên Kaggle

1. Tạo notebook mới hoặc **Upload** file `notebook_drmd_lamda_unified_v14.ipynb`.
2. Vào **Settings**:
   - Accelerator → **GPU** (bắt buộc)
   - Internet → On (lần đầu) hoặc Off nếu đã gắn đủ dataset
3. **Add Input**:
   - `thanhngo1007/drmd-lamda-dataset`
   - `thanhngo1007/lamda-full-processed`
   - (Tùy chọn) dataset chứa checkpoint zip
4. Chạy **Run All** hoặc chạy từng ô theo thứ tự.
5. Kết quả nằm ở `/kaggle/working/results/...`:
   - `tables/` — bảng tổng hợp, so sánh ghép cặp
   - `figures/` — hình (khi đủ dữ liệu)
   - `*_checkpoint_raw.zip` — checkpoint để dùng phiên sau
   - `*_full_test_report_bundle.zip` — gói báo cáo khi đủ 50/50

Thời gian tham khảo: khoảng 4–6 phút / lượt DRMD trên GPU Kaggle.  
50 lượt thường cần nhiều phiên. Notebook tự dừng theo `SESSION_LAUNCH_CUTOFF_HOURS` (mặc định 10,5 giờ) và xuất checkpoint.

---

## 3. Chạy trên Google Colab

1. Upload `notebook_drmd_lamda_unified_v14.ipynb` lên Colab.
2. Runtime → Change runtime type → **GPU**.
3. Tạo secret (biểu tượng chìa khóa bên trái):
   - Tên: `KAGGLE_API_TOKEN`
   - Giá trị: token API Kaggle của bạn (không dán token vào code)
4. (Tùy chọn) Mount Google Drive nếu muốn sao lưu checkpoint tự động.
5. Chạy **Ô 1**. Notebook sẽ:
   - Cài `kagglehub` nếu thiếu
   - Tải 2 dataset bắt buộc vào `/content/drmd_colab/input`
   - Dùng `/content/drmd_colab/working` làm thư mục ghi
6. Chạy tiếp các ô còn lại theo thứ tự.

**Checkpoint tùy chọn trên Colab:**
- Đặt trong Ô 1: `COLAB_CHECKPOINT_DATASET = "owner/slug"`
- Hoặc tạo secret `KAGGLE_CHECKPOINT_DATASET` với giá trị tương tự

Nếu đã mount Drive, checkpoint được copy vào:
`/content/drive/MyDrive/DRMD_checkpoints/`

---

## 4. Thứ tự các ô trong notebook

| Ô | Việc làm |
|---|---|
| 1 | Cấu hình, seed, ma trận phương pháp, bootstrap Colab |
| 2 | Khôi phục checkpoint (nếu có) |
| 3 | Kiểm kê artifact đã khôi phục |
| 4 | Bắt `UndefinedMetricWarning` ghi ra file |
| 5 | Cố định random / NumPy / PyTorch / cuDNN |
| 6 | Cài dependency + vá runtime DRMD |
| 7 | Nạp LAMDA, kiểm SHA-256 và split |
| 8 | Huấn luyện Static-MLP (10 seed) |
| 9 | Huấn luyện IRAL, IRAAL, FN, FN-BHR (40 lượt) |
| 10 | Tổng hợp monthly, summary, so sánh ghép cặp |
| 11 | Kiểm tra điều kiện đọc kết quả |
| 12 | Đóng gói zip bảng / checkpoint |

**Không đảo thứ tự ô.** Ô sau phụ thuộc biến và artifact của ô trước.

---

## 5. Tham số chính (đã khóa trong notebook)

| Tham số | Giá trị |
|---|---|
| Ngân sách B | 15 |
| Chi phí từ chối | −0,1 |
| mp | 1 |
| Bộ nhớ fine-tune | 5.000 mẫu |
| Số chu kỳ kiểm thử | 110 |
| Số mẫu test | 885.947 |
| Seed | `[0, 1, 7, 13, 26, 42, 73, 2026, 314159, 281083886]` |
| Cửa sổ thời gian | 12 tháng (`calendar_12`) |

Ba chỉnh sửa quan trọng của bản V14 (so với FN cũ):

1. Phạt FN chỉ áp dụng khi mô hình phân loại trực tiếp. Phần thưởng khi từ chối giữ nguyên như DRMD gốc.
2. Bộ điều khiển FN dùng hậu nghiệm Beta, chỉ đổi mức phạt khi khoảng tin cậy đủ hỗ trợ.
3. Ở FN-BHR, tỷ lệ phát lại mẫu FN giảm theo căn bậc hai của hệ số phạt.

---

## 6. Tiếp tục phiên (checkpoint)

Trong Ô 1 có biến:

```python
KAGGLE_RUN_PHASE = "phase_a_primary"   # huấn luyện
# KAGGLE_RUN_PHASE = "aggregate_only"  # chỉ tổng hợp từ artifact đã có
```

- Muốn tiếp tục huấn luyện: gắn checkpoint zip từ phiên trước, giữ `phase_a_primary`.
- Muốn chỉ tổng hợp lại bảng: đặt `aggregate_only` và gắn đủ artifact.

Notebook kiểm tra chữ ký và `protocol_version`. Không trộn artifact từ giao thức khác.

---

## 7. Đọc kết quả

Sau Ô 10, xem các file trong thư mục results:

| File | Ý nghĩa |
|---|---|
| `unified_integrity_summary.json` | `report_ready`, số lượt hợp lệ, lỗi |
| `unified_all_methods_overview.csv` | Trung bình ± sd theo phương pháp |
| `unified_paired_effects_summary.csv` | Hiệu ghép cặp (FN − IRAAL, FN-BHR − FN, …) |

**Chỉ diễn giải khi `report_ready = true`** (đủ 50/50).

---

## 8. Lưu ý quan trọng

- GPU bắt buộc khi huấn luyện (`REQUIRE_CUDA = True`).
- Không sửa `EXPECTED_LAMDA_FILE_SHA256` trừ khi cố ý đổi bộ dữ liệu.
- Notebook tự nhúng mã BHR và các vá runtime, không cần package `experiments` riêng.
- Trên Kaggle Dataset `drmd-lamda-dataset`, cấu trúc thư mục gốc cần có `references/DRMD` để notebook tìm project.

---

## Liên kết

- LAMDA gốc: [Hugging Face](https://huggingface.co/datasets/IQSeC-Lab/LAMDA)
- Dataset mã nguồn: [thanhngo1007/drmd-lamda-dataset](https://www.kaggle.com/datasets/thanhngo1007/drmd-lamda-dataset)
- Dataset parquet: [thanhngo1007/lamda-full-processed](https://www.kaggle.com/datasets/thanhngo1007/lamda-full-processed)
