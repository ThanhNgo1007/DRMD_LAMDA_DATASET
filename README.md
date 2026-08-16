# DRMD trên LAMDA — Hướng dẫn chạy (V15)

Notebook huấn luyện thống nhất cho thí nghiệm DRMD trên bộ dữ liệu LAMDA.
Chạy được trên **Kaggle** và **Google Colab**.

File notebook: `notebook_drmd_lamda_v15.ipynb`

## Giao thức V15

| Pha | Phương pháp | Số lượt |
|---|---|---|
| `phase_1_baselines` | Static-MLP, DRMD-IRAL, DRMD-IRAAL | 30 |
| `phase_2_proposed` | DRMD-FN, DRMD-FN-BHR, DRMD-FFCR | 30 |
| `aggregate_only` | Chỉ tổng hợp; không fit mô hình | 0 |

| Phương pháp | Mô tả |
|---|---|
| Static-MLP | MLP cố định trên cửa sổ huấn luyện ban đầu, không cập nhật |
| DRMD-IRAL | Phản hồi nội sinh $Q_t = R_t$ |
| DRMD-IRAAL | Cập nhật theo ngân sách $B = 15$ |
| DRMD-FN | IRAAL + phạt âm tính giả thích nghi (hậu nghiệm Beta) + cổng trôi dạt chưa gán nhãn |
| DRMD-FN-BHR | IRAAL + phạt FN thích nghi + hạn ngạch phát lại (BHR) thích nghi theo rủi ro kiểm toán; **không** ghép cổng trôi dạt |
| DRMD-FFCR | Full-feedback reference (mọi mẫu đều có nhãn sau khi khóa dự đoán) |

Mỗi phương pháp chạy **10 seed** → **60 lượt** khi đủ cả hai pha.
Báo cáo chỉ hợp lệ khi đủ **60/60** khóa `(phương pháp, seed)`.

Hai chiến lược đề xuất tuân thủ quan hệ nhân quả: dự đoán chu kỳ $t$ được khóa trước; điểm trôi dạt chỉ đọc $X_t$; nhãn kiểm toán của $t$ và hệ số/quota mới chỉ tác động từ lần fit ở $t+1$. Notebook không dùng họ mã độc, mức độ nguy hại hoặc trường VirusTotal để điều khiển phần thưởng.

---

## 1. Nguồn chính thống (bắt buộc đọc trước)

Notebook **không** đi kèm mã DRMD hay dữ liệu LAMDA. Bạn phải tự lấy từ nguồn gốc, rồi đóng gói thành 2 Kaggle Dataset theo đúng cấu trúc bên dưới.

### 1.1. Mã nguồn DRMD và phụ thuộc

| Thành phần | Nguồn chính thống | Ghi chú |
|---|---|---|
| DRMD | [github.com/s2labres/DRMD](https://github.com/s2labres/DRMD) | Paper: [arXiv:2508.18839](https://arxiv.org/abs/2508.18839) |
| Tesseract (temporal split) | [github.com/s2labres/tesseract-ml-release](https://github.com/s2labres/tesseract-ml-release) | Thư viện chia train/test theo thời gian |
| RPAL | [github.com/s2labres/RPAL](https://github.com/s2labres/RPAL) | Phụ thuộc liên quan active learning |

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
    ├── DRMD/
    ├── tesseract-ml-release/
    └── RPAL/
```

Đóng gói thư mục `DRMD_LAMDA_DATASET` thành Kaggle Dataset.
Notebook tìm project bằng cách kiểm tra sự tồn tại của `references/DRMD`.

### 1.2. Dữ liệu LAMDA (đặc trưng đã xử lý)

| Thành phần | Nguồn chính thống |
|---|---|
| LAMDA | [Hugging Face: IQSeC-Lab/LAMDA](https://huggingface.co/datasets/IQSeC-Lab/LAMDA) |
| Paper / repo | [github.com/iqsec-lab/lamda](https://github.com/iqsec-lab/lamda) |

Notebook khóa một revision cụ thể và kiểm SHA-256 từng file parquet.
Revision trong notebook: `ad9614bdd5556767f97ced2fce797c2f06408ebf`.

Gợi ý tải:

```bash
# Cần: pip install huggingface_hub
huggingface-cli download IQSeC-Lab/LAMDA \
  --repo-type dataset \
  --revision ad9614bdd5556767f97ced2fce797c2f06408ebf \
  --local-dir ./lamda_raw
```

Cấu trúc notebook mong đợi:

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
Nếu lệch, hoặc cập nhật dict SHA (khi cố ý đổi revision), hoặc dùng đúng bản đã khóa.

### 1.3. Dataset tham khảo (đã đóng gói sẵn)

Nếu muốn bỏ qua bước tự đóng gói, có thể dùng bản đã chuẩn bị sẵn (cùng cấu trúc và SHA đã khóa):

| Dataset Kaggle | Vai trò |
|---|---|
| [thanhngo1007/drmd-lamda-dataset](https://www.kaggle.com/datasets/thanhngo1007/drmd-lamda-dataset) | Tham khảo: mã nguồn + `references/DRMD` |
| [thanhngo1007/lamda-full-processed](https://www.kaggle.com/datasets/thanhngo1007/lamda-full-processed) | Tham khảo: parquet Baseline theo năm |

Đây chỉ là **bản tiện lợi**. Nguồn chính thống vẫn là GitHub s2labres và Hugging Face IQSeC-Lab.
Khi tái tạo thí nghiệm hoặc xuất bản, nên ghi rõ bạn đã lấy từ nguồn gốc.

### 1.4. Checkpoint (tùy chọn)

Nếu đã có `*_checkpoint_raw.zip` từ phiên trước, tạo thêm một Kaggle Dataset chứa file zip đó.
Notebook sẽ khôi phục và bỏ qua các lượt đã hoàn tất (cùng `protocol_version` V15).

---

## 2. Chạy trên Kaggle

1. Tạo notebook mới hoặc **Upload** file `notebook_drmd_lamda_v15.ipynb`.
2. Vào **Settings**:
   - Accelerator → **GPU** (bắt buộc khi huấn luyện)
   - Internet → On (lần đầu) hoặc Off nếu đã gắn đủ dataset
3. **Add Input**:
   - Dataset mã nguồn (có `references/DRMD`)
   - Dataset parquet LAMDA (có thư mục `Baseline/`)
   - (Tùy chọn) dataset checkpoint
4. Trong Ô 1, kiểm tra:
   - `RUNTIME_PLATFORM = "kaggle"` (mặc định; hoặc `"auto"`)
   - `KAGGLE_RUN_PHASE` theo pha cần chạy (xem mục 6)
5. Chạy **Run All** hoặc chạy từng ô theo thứ tự.
6. Kết quả nằm ở `/kaggle/working/results/...`:
   - `tables/` — bảng tổng hợp, so sánh ghép cặp, Holm
   - `figures/` — hình (khi đủ dữ liệu)
   - `*_checkpoint_raw.zip` — checkpoint để dùng phiên sau
   - `*_full_test_report_bundle.zip` — gói báo cáo khi đủ 60/60

Thời gian tham khảo: khoảng vài phút mỗi lượt DRMD trên GPU Kaggle.
60 lượt thường cần nhiều phiên. Notebook xuất checkpoint cuối mỗi phiên huấn luyện.

---

## 3. Chạy trên Google Colab

1. Upload `notebook_drmd_lamda_v15.ipynb` lên Colab.
2. Runtime → Change runtime type → **GPU**.
3. Trong Ô 1 đặt:
   ```python
   RUNTIME_PLATFORM = "colab"
   ```
4. Tạo secret (biểu tượng chìa khóa):
   - Tên: `KAGGLE_API_TOKEN`
   - Giá trị: token API Kaggle (không dán token vào mã nguồn)
5. (Khuyến nghị) Mount Google Drive để sao lưu checkpoint tự động vào
   `/content/drive/MyDrive/DRMD_checkpoints/`.
6. Nếu dùng dataset tự đóng gói, sửa handle `owner/slug` trong
   `COLAB_REQUIRED_KAGGLE_DATASETS`; mặc định notebook trỏ tới bản tham khảo.
7. Chạy **Ô 1**. Notebook sẽ:
   - Cài `kagglehub` nếu thiếu
   - Tải dataset vào `/content/drmd_colab/input`
   - Dùng `/content/drmd_colab/working` làm thư mục ghi
8. Chạy tiếp các ô còn lại theo thứ tự.

**Checkpoint tùy chọn trên Colab:**

- Đặt trong Ô 1: `COLAB_CHECKPOINT_DATASET = "owner/slug"`
- Hoặc tạo secret `KAGGLE_CHECKPOINT_DATASET`

---

## 4. Thứ tự các ô trong notebook

| Ô | Việc làm |
|---|---|
| 1 | Cấu hình, seed, ma trận phương pháp, bootstrap runtime |
| 2 | Khôi phục checkpoint (nếu có) |
| 3 | Kiểm kê artifact đã khôi phục |
| 4 | Bắt `UndefinedMetricWarning` ghi ra file |
| 5 | Cố định random / NumPy / PyTorch / cuDNN |
| 6 | Cài dependency + vá runtime DRMD |
| 7 | Nạp LAMDA, kiểm SHA-256 và split thời gian |
| 8 | Huấn luyện Static-MLP (10 seed) |
| 9 | Huấn luyện các chiến lược DRMD theo pha đã chọn |
| 10 | Tổng hợp monthly, summary, so sánh ghép cặp, Holm |
| 11 | Điều kiện đọc kết quả (`REPORT_READY`) |
| 12 | Đóng gói zip bảng / checkpoint |

**Không đảo thứ tự ô.** Ô sau phụ thuộc biến và artifact của ô trước.

---

## 5. Tham số chính (đã khóa trong notebook)

| Tham số | Giá trị |
|---|---|
| Ngân sách $B$ | 15 |
| Chi phí từ chối $c_R$ | −0,1 |
| $m_p$ | 1 |
| Bộ nhớ fine-tune | 5.000 mẫu |
| Số chu kỳ kiểm thử | 110 |
| Số mẫu test | 885.947 |
| Seed | `[0, 1, 7, 13, 26, 42, 73, 2026, 314159, 281083886]` |
| Cửa sổ thời gian | 12 tháng (`calendar_12`) |
| Revision LAMDA | `ad9614bdd5556767f97ced2fce797c2f06408ebf` |
| Protocol version | `drmd_lamda_fn_bhr_unified_v15_2026-08-15` |

### Cơ chế quan trọng

1. **Phạt FN** chỉ áp dụng khi mô hình phân loại trực tiếp. Phần thưởng hành động từ chối giữ nguyên như DRMD gốc.
2. **Bộ điều khiển FN** dùng hậu nghiệm Beta (Jeffreys); chỉ đổi $\lambda$ khi khoảng tin cậy đủ hỗ trợ; hiệu lực từ chu kỳ $t+1$.
3. **DRMD-FN**: thêm cổng trôi dạt trên đặc trưng chưa gán nhãn $X_t$ (chiếu ngẫu nhiên cố định, cửa sổ quá khứ, phân vị ngưỡng đã khóa).
4. **DRMD-FN-BHR**: hạn ngạch phát lại thích nghi theo rủi ro kiểm toán (hậu nghiệm FN/FP, hỗ trợ lớp, sức chứa bộ nhớ); **không** ghép cổng trôi dạt với BHR trong cùng một phương pháp.
5. Nhãn của chu kỳ $t$ chỉ được dùng khi fit từ chu kỳ $t+1$.

---

## 6. Tiếp tục phiên (checkpoint)

Trong Ô 1:

```python
KAGGLE_RUN_PHASE = "phase_1_baselines"   # 30 lượt: Static, IRAL, IRAAL
# KAGGLE_RUN_PHASE = "phase_2_proposed"  # 30 lượt: FN, FN-BHR, FFCR
# KAGGLE_RUN_PHASE = "aggregate_only"    # chỉ tổng hợp khi đã đủ artifact
```

- Tiếp tục huấn luyện: gắn checkpoint zip từ phiên trước, giữ đúng pha.
- Chỉ tổng hợp bảng: đặt `aggregate_only` và gắn đủ artifact.
- Notebook kiểm tra chữ ký và `protocol_version`. **Không trộn** artifact từ giao thức khác.

Checkpoint zip gồm `raw/static_baselines/` và `raw/<STRATEGY_RUN_TAG>/`.
Cuối ô Static và cuối ô chiến lược đều xuất zip; trên Colab có thể sao lưu Drive nếu đã mount.

---

## 7. Đọc kết quả

Sau Ô 10, xem thư mục results:

| File | Ý nghĩa |
|---|---|
| summary / integrity JSON | `REPORT_READY`, số lượt hợp lệ, lỗi |
| `v15_all_methods_summary_by_seed.csv` | AUT / pooled theo method–seed |
| `v15_paired_effects_summary.csv` | Hiệu ghép cặp trung bình |
| `v15_primary_paired_tests_holm.csv` | KTC 95%, paired-t, Wilcoxon, Holm |

**Chỉ diễn giải khi `REPORT_READY = True`** (đủ 60/60).

Các đối chứng chính (cùng 10 seed):

- IRAAL − IRAL
- FN − IRAAL
- FN-BHR − FN
- FN-BHR − IRAAL

---

## 8. Lưu ý quan trọng

- GPU bắt buộc khi huấn luyện.
- Không sửa `EXPECTED_LAMDA_FILE_SHA256` trừ khi cố ý đổi bộ dữ liệu / revision.
- Notebook tự nhúng mã BHR, FN controller, drift gate và các vá runtime.
- Khi công bố kết quả, trích dẫn paper DRMD và paper/dataset LAMDA;
  dataset Kaggle đóng gói sẵn chỉ là tiện ích tái hiện.

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

## Repository

- GitHub: [ThanhNgo1007/DRMD_LAMDA_DATASET](https://github.com/ThanhNgo1007/DRMD_LAMDA_DATASET)
- Notebook nộp: `notebook_drmd_lamda_v15.ipynb`
