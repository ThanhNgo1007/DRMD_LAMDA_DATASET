# DRMD trên LAMDA — Hướng dẫn chạy

Notebook huấn luyện thống nhất cho 6 phương pháp trên bộ dữ liệu LAMDA.  
Chạy được trên **Kaggle** và **Google Colab**.

| Phương pháp | Mô tả |
|---|---|
| Static-MLP | MLP cố định trên cửa sổ huấn luyện ban đầu |
| DRMD-IRAL | Phản hồi nội sinh \(Q_t = R_t\) |
| DRMD-IRAAL | Cập nhật theo ngân sách B = 15 |
| DRMD-FN | IRAAL + phạt false-negative thích nghi |
| DRMD-FN-BHR | DRMD-FN + Balanced Hard Replay |
| DRMD-FFCR | Full-feedback reference (mọi mẫu đều có nhãn) |

Mỗi phương pháp chạy 10 seed → **60 lượt**. Báo cáo chỉ hợp lệ khi đủ 60/60.

---

## 1. Nguồn chính thống (bắt buộc đọc trước)

Notebook **không** đi kèm mã DRMD hay dữ liệu LAMDA. Bạn phải tự lấy từ nguồn gốc, rồi đóng gói thành 2 Kaggle Dataset theo đúng cấu trúc bên dưới.

### 1.1. Mã nguồn DRMD và phụ thuộc

| Thành phần | Nguồn chính thống | Ghi chú |
|---|---|---|
| DRMD | [github.com/s2labres/DRMD](https://github.com/s2labres/DRMD) | Paper: [arXiv:2508.18839](https://arxiv.org/abs/2508.18839) |
| Tesseract (temporal split) | [github.com/s2labres/tesseract-ml-release](https://github.com/s2labres/tesseract-ml-release) | Thư viện chia train/test theo thời gian |
| RPAL | [github.com/s2labres/RPAL](https://github.com/s2labres/RPAL) | Phụ thuộc liên quan active learning / recovery |

Cách lấy (ví dụ):

```bash
mkdir -p DRMD_LAMDA_DATASET/references
cd DRMD_LAMDA_DATASET/references

git clone https://github.com/s2labres/DRMD.git
git clone https://github.com/s2labres/tesseract-ml-release.git
git clone https://github.com/s2labres/RPAL.git
```

Cấu trúc tối thiểu sau khi clone:

```text
DRMD_LAMDA_DATASET/
└── references/
    ├── DRMD/                    # mã gốc DRMD
    ├── tesseract-ml-release/    # thư viện temporal
    └── RPAL/                    # phụ thuộc
```

Đóng gói thư mục `DRMD_LAMDA_DATASET` thành Kaggle Dataset (ví dụ tên `yourname/drmd-lamda-dataset`).  
Notebook tìm project bằng cách kiểm tra sự tồn tại của `references/DRMD`.

### 1.2. Dữ liệu LAMDA (đặc trưng đã xử lý)

| Thành phần | Nguồn chính thống |
|---|---|
| LAMDA | [Hugging Face: IQSeC-Lab/LAMDA](https://huggingface.co/datasets/IQSeC-Lab/LAMDA) |
| Paper / repo | [github.com/iqsec-lab/lamda](https://github.com/iqsec-lab/lamda) · ICLR 2026 |

Notebook khóa một revision cụ thể và kiểm SHA-256 từng file parquet.  
Revision đang dùng trong notebook: `ad9614bdd5556767f97ced2fce797c2f06408ebf` (có thể xem trong ô cấu hình).

Gợi ý tải:

```bash
# Cần: pip install huggingface_hub
huggingface-cli download IQSeC-Lab/LAMDA --repo-type dataset --revision ad9614bdd5556767f97ced2fce797c2f06408ebf --local-dir ./lamda_raw
```

Sau khi tải, tổ chức (hoặc chuyển đổi) thành cấu trúc mà notebook mong đợi:

```text
lamda-full-processed/
└── Baseline/
    ├── 2013/
    │   ├── 2013_train.parquet
    │   └── 2013_test.parquet
    ├── 2014/
    │   └── ...
    └── ...
```

Mỗi file phải khớp SHA-256 trong biến `EXPECTED_LAMDA_FILE_SHA256` của notebook.  
Nếu bạn tự xử lý từ bản gốc Hugging Face (ví dụ chọn variant Baseline), hãy chạy ô kiểm tra SHA trong notebook; nếu lệch, hoặc cập nhật dict SHA, hoặc dùng đúng bản đã khóa.

Đóng gói thư mục `lamda-full-processed` (có `Baseline/`) thành Kaggle Dataset thứ hai.

### 1.3. Dataset tham khảo (đã đóng gói sẵn)

Nếu muốn bỏ qua bước tự đóng gói, có thể dùng bản đã chuẩn bị sẵn (cùng cấu trúc và SHA đã khóa):

| Dataset Kaggle | Vai trò |
|---|---|
| [thanhngo1007/drmd-lamda-dataset](https://www.kaggle.com/datasets/thanhngo1007/drmd-lamda-dataset) | Tham khảo: mã nguồn + `references/DRMD` |
| [thanhngo1007/lamda-full-processed](https://www.kaggle.com/datasets/thanhngo1007/lamda-full-processed) | Tham khảo: parquet Baseline theo năm |

Đây chỉ là **bản tiện lợi**. Nguồn chính thống vẫn là GitHub s2labres và Hugging Face IQSeC-Lab như trên. Khi tái tạo thí nghiệm hoặc xuất bản, nên ghi rõ bạn đã lấy từ nguồn gốc.

### 1.4. Checkpoint (tùy chọn)

Nếu đã có `*_checkpoint_raw.zip` từ phiên trước, tạo thêm một Kaggle Dataset chứa file zip đó. Notebook sẽ khôi phục và bỏ qua các lượt đã hoàn tất.

---

## 2. Chạy trên Kaggle

1. Tạo notebook mới hoặc **Upload** file `notebook_drmd_lamda_unified_v14.ipynb`.
2. Vào **Settings**:
   - Accelerator → **GPU** (bắt buộc)
   - Internet → On (lần đầu) hoặc Off nếu đã gắn đủ dataset
3. **Add Input**:
   - Dataset mã nguồn (cấu trúc có `references/DRMD`) — của bạn hoặc bản tham khảo
   - Dataset parquet LAMDA (có thư mục `Baseline/`)
   - (Tùy chọn) dataset checkpoint
4. Chạy **Run All** hoặc chạy từng ô theo thứ tự.
5. Kết quả nằm ở `/kaggle/working/results/...`:
   - `tables/` — bảng tổng hợp, so sánh ghép cặp
   - `figures/` — hình (khi đủ dữ liệu)
   - `*_checkpoint_raw.zip` — checkpoint để dùng phiên sau
   - `*_full_test_report_bundle.zip` — gói báo cáo khi đủ 60/60

Thời gian tham khảo: khoảng 4–6 phút / lượt DRMD trên GPU Kaggle.  
60 lượt thường cần nhiều phiên. Notebook tự dừng theo `SESSION_LAUNCH_CUTOFF_HOURS` (mặc định 10,5 giờ) và xuất checkpoint.

---

## 3. Chạy trên Google Colab

1. Upload `notebook_drmd_lamda_unified_v14.ipynb` lên Colab.
2. Runtime → Change runtime type → **GPU**.
3. Tạo secret (biểu tượng chìa khóa bên trái):
   - Tên: `KAGGLE_API_TOKEN`
   - Giá trị: token API Kaggle của bạn (không dán token vào code)
4. (Tùy chọn) Mount Google Drive nếu muốn sao lưu checkpoint tự động.
5. Trong Ô 1, nếu dùng dataset tự đóng gói, sửa handle cho đúng `owner/slug` của bạn; mặc định notebook trỏ tới bản tham khảo.
6. Chạy **Ô 1**. Notebook sẽ:
   - Cài `kagglehub` nếu thiếu
   - Tải dataset vào `/content/drmd_colab/input`
   - Dùng `/content/drmd_colab/working` làm thư mục ghi
7. Chạy tiếp các ô còn lại theo thứ tự.

**Checkpoint tùy chọn trên Colab:**
- Đặt trong Ô 1: `COLAB_CHECKPOINT_DATASET = "owner/slug"`
- Hoặc tạo secret `KAGGLE_CHECKPOINT_DATASET`

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
| 9 | Huấn luyện IRAL, IRAAL, FN, FN-BHR, FFCR (50 lượt) |
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
| Revision LAMDA | `ad9614bdd5556767f97ced2fce797c2f06408ebf` |

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
| `unified_paired_effects_summary.csv` | Hiệu ghép cặp (FN − IRAAL, FN-BHR − FN, FFCR − …) |

**Chỉ diễn giải khi `report_ready = true`** (đủ 60/60).

---

## 8. Lưu ý quan trọng

- GPU bắt buộc khi huấn luyện (`REQUIRE_CUDA = True`).
- Không sửa `EXPECTED_LAMDA_FILE_SHA256` trừ khi cố ý đổi bộ dữ liệu / revision.
- Notebook tự nhúng mã BHR và các vá runtime; không cần package `experiments` có sẵn trên Dataset.
- Khi công bố kết quả, trích dẫn đúng paper DRMD và paper LAMDA; dataset Kaggle đóng gói sẵn chỉ là tiện ích tái hiện.

---

## Liên kết nguồn chính thống

- DRMD (code): [github.com/s2labres/DRMD](https://github.com/s2labres/DRMD)
- DRMD (paper): [arXiv:2508.18839](https://arxiv.org/abs/2508.18839)
- Tesseract: [github.com/s2labres/tesseract-ml-release](https://github.com/s2labres/tesseract-ml-release)
- RPAL: [github.com/s2labres/RPAL](https://github.com/s2labres/RPAL)
- LAMDA (dataset): [Hugging Face IQSeC-Lab/LAMDA](https://huggingface.co/datasets/IQSeC-Lab/LAMDA)
- LAMDA (repo): [github.com/iqsec-lab/lamda](https://github.com/iqsec-lab/lamda)

## Dataset tham khảo (đã đóng gói)

- [thanhngo1007/drmd-lamda-dataset](https://www.kaggle.com/datasets/thanhngo1007/drmd-lamda-dataset)
- [thanhngo1007/lamda-full-processed](https://www.kaggle.com/datasets/thanhngo1007/lamda-full-processed)
