# Environmental Sound Classification on ESC-50 Using a Multi-Task CNN with Hierarchical Labels

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: ESC-50](https://img.shields.io/badge/Dataset-ESC--50-green.svg)](https://github.com/karolpiczak/ESC-50)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TinyML / Edge AI](https://img.shields.io/badge/Target-IoT%20%2F%20Edge%20AI-orange.svg)]()

---

## 📌 Thông tin đề tài

* **Tên đề tài:** Phân loại Âm thanh Môi trường trên ESC-50 bằng CNN Đa nhiệm với Nhãn Phân cấp (50 Lớp Chi tiết & 5 Nhóm lớn)  
* **Tiếng Anh:** *Environmental Sound Classification on ESC-50 Using a Multi-Task CNN with 50 Fine Classes and 5 Coarse Groups*  
* **Học phần:** Trí tuệ Nhân tạo cho IoT (AI for IoT)  
* **Đơn vị:** Khoa Công nghệ Thông tin — Trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh (HCMUTE)  
* **Giảng viên hướng dẫn:** ThS. Hồ Nhựt Minh  
* **Sinh viên thực hiện:** Dương Minh Duy — MSSV: 23110083  

---

## 📖 1. Giới thiệu & Đặt vấn đề

Phân loại âm thanh môi trường (**Environmental Sound Classification - ESC**) là bài toán cốt lõi trong các hệ thống IoT thông minh (giám sát an ninh, phát hiện tai nạn, đô thị thông minh, bảo tồn sinh thái). Tuy nhiên, triển khai trên thiết bị nhúng/Edge AI đòi hỏi mô hình phải vừa **nhẹ**, vừa **tiết kiệm bộ nhớ**, đồng thời duy trì độ chính xác và tính nhất quán giữa các mức ngữ nghĩa.

Đề tài khảo sát việc xây dựng kiến trúc **Convolutional Neural Network (CNN) đa nhiệm** với bộ trích xuất đặc trưng dùng chung (**Shared Backbone**) để cùng lúc dự đoán:
1. **50 lớp chi tiết (Fine classes)**.
2. **5 nhóm lớn ngữ nghĩa (Coarse groups)**.

Đồng thời, đề tài đánh giá biến thể mở rộng **M2** bổ sung hàm mất mát phân kỳ Jensen-Shannon (**JSD Consistency Loss**) nhằm tối ưu hóa sự tương thích phân phối xác suất giữa hai nhánh.

```
                    ┌─────────────────────────┐
                    │ Input Audio (5s, 16kHz) │
                    └────────────┬────────────┘
                                 │ Log-Mel Spectrogram (64 x 251)
                                 ▼
                    ┌─────────────────────────┐
                    │     Shared Backbone     │
                    │  (4 Conv Blocks, 83.7k) │
                    └────────────┬────────────┘
                                 │ Feature Vector h (96-D)
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌───────────────────────┐ ┌───────────────────────┐
        │       Fine Head       │ │      Coarse Head      │
        │ Linear (96 -> 50)     │ │ Linear (96 -> 5)      │
        └───────────┬───────────┘ └───────────┬───────────┘
                    │ p_f                     │ p_c
                    ▼                         ▼
             50 Fine Classes            5 Coarse Groups
                    │                         │
                    └───────► [JSD Loss] ◄────┘ (Biến thể M2)
```

---

## 🗂️ 2. Tập dữ liệu & Cấu trúc Phân cấp

Đề tài sử dụng bộ dữ liệu chuẩn quốc tế **ESC-50** (Piczak, 2015):
* **Quy mô:** 2.000 file âm thanh WAV chất lượng cao (5 giây/file), cân bằng 40 đoạn/lớp trên 50 lớp.
* **5 Nhóm lớn (Coarse Groups):**
  1. `Animals` (Động vật - Lớp 0 đến 9)
  2. `Natural soundscapes & water sounds` (Âm thanh tự nhiên & nước - Lớp 10 đến 19)
  3. `Human, non-speech sounds` (Âm thanh con người phi ngôn ngữ - Lớp 20 đến 29)
  4. `Domestic/interior sounds` (Âm thanh trong nhà/nội thất - Lớp 30 đến 39)
  5. `Exterior/urban noises` (Âm thanh đường phố/ngoại thất - Lớp 40 đến 49)

### Ánh xạ toán học & Tính nhất quán phân cấp (HCR)
Ánh xạ từ nhãn chi tiết $k \in \{0, \dots, 49\}$ sang nhóm lớn $c \in \{0, \dots, 4\}$ được xác định bởi:
$$g(k) = \lfloor k / 10 \rfloor$$
Ma trận phân cấp $M \in \mathbb{R}^{5 \times 50}$ với $M_{c,k} = 1$ khi $g(k) = c$ và $0$ nếu ngược lại.

**Chỉ số Nhất quán Phân cấp Thô (Raw Hierarchical Consistency Rate - HCR):**
$$\text{HCR} = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} \mathbb{I}\left[ g(\hat{y}_{f, i}) = \hat{y}_{c, i} \right]$$

---

## ⚙️ 3. Phương pháp & Kiến trúc Mô hình

### Tiền xử lý & Trích xuất Đặc trưng
* **Sampling Rate:** $16.000\text{ Hz}$ (được chọn tối ưu cho hệ thống IoT/nhúng).
* **Biểu diễn:** Log-Mel Spectrogram với $N_{mel} = 64$, $N_{FFT} = 512$, $\text{hop\_length} = 320$, $\text{win\_length} = 512$, $\text{center} = \text{True} \implies$ Kích thước tensor: **$1 \times 64 \times 251$**.
* **Chuẩn hóa (Normalization):** Tính toán $\mu_{train}, \sigma_{train}$ trên từng dải mel **chỉ từ tập Train** (Zero Data-Leakage).
* **Tăng cường dữ liệu:** SpecAugment rút gọn (Frequency masking $\leq 8$ bands, Time masking $\leq 25$ frames) chỉ áp dụng khi huấn luyện.

### Danh mục các mô hình nghiên cứu
1. **B0 (Baseline MFCC-SVM):** 40 hệ số MFCC $\to$ Vector đặc trưng 80 chiều (mean & std theo thời gian) $\to$ StandardScaler $\to$ SVM RBF ($C=10$).
2. **B1 (Baseline Single-Task CNN):** Shared Backbone (4 khối Conv2D: 16-32-64-96 kênh, BatchNorm2d, ReLU, Stride=2) $\to$ Global Average Pooling $\to$ Dropout(0.3) $\to$ Fine Head (96 $\to$ 50).
3. **M1 (Multi-Task CNN):** Cùng Backbone B1 $\to$ Fine Head (96 $\to$ 50) & Coarse Head (96 $\to$ 5).  
   Hàm mất mát: $\mathcal{L}_{total} = \mathcal{L}_{fine} + 0.3 \cdot \mathcal{L}_{coarse}$.
4. **M2 (Multi-Task CNN + JSD Consistency):** Kiến trúc M1 bổ sung thành phần phân kỳ Jensen-Shannon giữa phân phối thô trực tiếp $p_c$ và phân phối thô gom cụm $Mp_f$:  
   $$\mathcal{L}_{total} = \mathcal{L}_{fine} + 0.3 \cdot \mathcal{L}_{coarse} + 0.1 \cdot \mathcal{L}_{JSD}(p_c \parallel Mp_f)$$

---

## 📊 4. Kết quả Thực nghiệm

Toàn bộ các mô hình được đánh giá nghiêm ngặt theo **giao thức 5-Fold Cross Validation chuẩn** với lịch chia cố định **3 Train / 1 Validation / 1 Test** (1.200 train / 400 val / 400 test mỗi round).

### Bảng 1: Kết quả Trung bình 5-Fold Cross Validation ($\text{Mean} \pm \text{Std}$)

| Mô hình | Số tham số | $Acc_{fine}$ (%) | $\text{Macro-F1}_{fine}$ (%) | $Acc_{coarse}^{map}$ (%) | $Acc_{coarse}^{head}$ (%) | $HCR_{raw}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (MFCC-SVM)** | — | $41.75 \pm 5.30$ | $39.81 \pm 5.04$ | $58.70 \pm 4.02$ | N/A | N/A |
| **B1 (Single-Task CNN)** | **83.746** | $56.70 \pm 3.75$ | $55.16 \pm 3.97$ | $69.40 \pm 2.53$ | N/A | N/A |
| **M1 (Multi-Task CNN)** | 84.231 | $56.45 \pm 6.05$ | $55.00 \pm 5.97$ | $69.85 \pm 4.14$ | $62.00 \pm 3.64$ | $71.70 \pm 2.57$ |
| **M2 (Multi-Task + JSD)** | 84.231 | **59.70 ± 3.39** | **58.21 ± 3.47** | **72.80 ± 3.14** | **62.90 ± 2.63** | **72.60 ± 1.76** |

### Bảng 2: Kết quả Đánh giá Gộp Out-Of-Fold (Toàn bộ 2.000 mẫu ESC-50)

| Mô hình | $Acc_{fine}$ (%) | $\text{Macro-F1}_{fine}$ (%) | $Acc_{coarse}^{map}$ (%) | $Acc_{coarse}^{head}$ (%) | $HCR_{raw}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B0 (MFCC-SVM)** | 41.75% | 40.53% | 58.70% | N/A | N/A |
| **B1 (Single-Task CNN)** | 56.70% | 55.84% | 69.40% | N/A | N/A |
| **M1 (Multi-Task CNN)** | 56.45% | 55.68% | 69.85% | 62.00% | 71.70% |
| **M2 (Multi-Task + JSD)** | **59.70%** | **58.84%** | **72.80%** | **62.90%** | **72.60%** |

> **Nhận xét chính:**
> * **M2** vượt trội toàn diện so với baseline B1 (+3.00% $Acc_{fine}$, +3.05% $\text{Macro-F1}$), chứng minh việc ràng buộc phân phối ngữ nghĩa thông qua Jensen-Shannon Divergence giúp không gian đặc trưng biểu diễn tốt hơn.
> * Sự bổ sung Coarse Head ở **M1/M2** chỉ tăng **485 tham số** (+0.58%), chi phí tính toán tăng không đáng kể nhưng đem lại khả năng phân loại phân cấp đa tầng.

---

## ⚡ 5. Phân tích Khả năng Triển khai IoT & TinyML

| Đặc tính phần cứng / tài nguyên | Giá trị đo đạc / Tính toán | Ý nghĩa thực tiễn trên Edge / MCU |
| :--- | :--- | :--- |
| **Số tham số mô hình (Parameters)** | **84.231** tham số | Cực kỳ nhỏ gọn so với các mô hình âm thanh chuẩn (ResNet, AST, PANNs) |
| **Kích thước Flash/ROM (Float32)** | **~336.9 KiB** (Weights) | Vừa vặn trong bộ nhớ Flash của vi điều khiển (ESP32-S3, STM32F4/F7) |
| **Kích thước Flash/ROM (INT8 Quantized)** | **~84.2 KiB** | Tối ưu hóa sâu cho TinyML / TensorFlow Lite for Microcontrollers |
| **Đầu vào Spectrogram RAM** | **62.75 KiB** ($64 \times 251 \times 4$ bytes) | Dễ dàng chứa trong SRAM |
| **Waveform Buffer RAM (5s, 16kHz)** | **312.5 KiB** ($80.000 \times 4$ bytes) | Có thể tối ưu hóa qua xử lý luồng (streaming sliding window) |
| **Real-Time Factor (RTF)** | **< 0.05** (trên Edge CPU/GPU) | Xử lý nhanh hơn 20 lần so với thời lượng thực của âm thanh |

---

## 📁 6. Cấu trúc Thư mục Dự án

```
AI_for_IOT/
├── DuongMinhDuy_final.docx          # 📄 Báo cáo tổng kết toàn văn học phần cuối kỳ
├── DuongMinhDuy_final.ipynb         # 📓 Jupyter Notebook hoàn chỉnh (EDA, 5-Fold Training B0-M2, Visualizations)
├── DuongMinhDuy_final.pptx          # 📊 Slide thuyết trình báo cáo đề tài
├── README.md                        # 📘 Tài liệu hướng dẫn & công bố kết quả nghiên cứu
├── .gitignore                       # ⚙️ Cấu hình loại trừ file cache & audio dung lượng lớn
│
├── src/                             # 🧱 Mã nguồn module cốt lõi
│   ├── mapping.py                   # Bảng tra 50 lớp -> 5 nhóm, hàm g(k), ma trận gom cụm M
│   └── models/                      # Định nghĩa kiến trúc PyTorch
│       ├── backbone.py              # Shared 4-Block CNN Backbone & JSD Loss
│       ├── single_task_cnn.py       # Mô hình đơn nhiệm B1
│       ├── multitask_cnn.py         # Mô hình đa nhiệm M1 & M2
│       └── baseline_svm.py          # Pipeline trích xuất MFCC 80D & SVM RBF
│
├── eda_and_training.py              # 🚀 Toàn bộ kịch bản thực thi EDA & Huấn luyện 5-Fold
├── generate_all_charts.py           # 📈 Kịch bản tự động xuất biểu đồ & ma trận nhầm lẫn
├── create_presentation.py           # 📑 Kịch bản sinh template slide trình bày
│
├── reports/                         # 📊 Kết quả thực nghiệm & Báo cáo
│   ├── comparison_all_models.csv    # Bảng tổng hợp số liệu 5-Fold
│   ├── comparison_oof_pooled.csv    # Bảng số liệu Out-of-fold 2.000 mẫu
│   ├── eda_figures/                 # Biểu đồ phân tích khám phá dữ liệu (Spectrogram, PCA, Waveform)
│   └── figures/                     # Đồ thị học, biểu đồ so sánh, Confusion Matrix B0-M2
│
└── experiments/                     # 💾 Nhật ký thực nghiệm & Checkpoints
    ├── manifests/                   # Danh sách phân chia 5-Fold chuẩn 3/1/1
    ├── stats/                       # Thống kê Mel μ, σ riêng theo từng vòng train
    ├── checkpoints/                 # Trọng số tốt nhất đã huấn luyện của B0, B1, M1, M2
    ├── predictions/                 # File CSV lưu kết quả dự đoán chi tiết Out-of-fold
    └── logs/                        # File log quá trình huấn luyện
```

---

## 🚀 7. Hướng dẫn Cài đặt & Tái lập Kết quả

### Bước 1: Clone kho lưu trữ
```bash
git clone https://github.com/Duyy1103/AI_for_IOT.git
cd AI_for_IOT
```

### Bước 2: Tạo môi trường & Cài đặt thư viện
Khuyến nghị sử dụng **Conda** hoặc **venv** với Python 3.10+:
```bash
conda create -n ai_iot python=3.10 -y
conda activate ai_iot

# Cài đặt PyTorch (CUDA nếu có)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện xử lý âm thanh & khoa học dữ liệu
pip install librosa scikit-learn numpy pandas matplotlib seaborn tqdm python-docx python-pptx
```

### Bước 3: Chuẩn bị dữ liệu ESC-50
Tải bộ dữ liệu [ESC-50](https://github.com/karolpiczak/ESC-50) và giải nén thư mục `audio` cùng file `meta/esc50.csv` vào `ESC-50-master/`:
```
ESC-50-master/
├── audio/          # Chứa 2.000 file .wav
└── meta/
    └── esc50.csv   # File nhãn metadata
```

### Bước 4: Chạy huấn luyện & Đánh giá 5-Fold
Bạn có thể mở và chạy toàn bộ từ đầu đến cuối trên Jupyter Notebook:
```bash
jupyter notebook DuongMinhDuy_final.ipynb
```
Hoặc thực thi trực tiếp qua Python script:
```bash
python eda_and_training.py
```

### Bước 5: Sinh lại toàn bộ biểu đồ báo cáo
```bash
python generate_all_charts.py
```

---

## 📦 8. Danh mục Sản phẩm Bàn giao

1. 📄 **`DuongMinhDuy_final.docx`**: Báo cáo học phần cuối kỳ hoàn chỉnh, chuẩn quy cách khoa học, trình bày toàn diện từ cơ sở lý thuyết, quy trình tiền xử lý, kiến trúc chi tiết, phân tích sai số đến bài toán triển khai Edge AI/IoT.
2. 📓 **`DuongMinhDuy_final.ipynb`**: File Notebook đầy đủ chứa toàn bộ mã nguồn, thực thi khám phá EDA, huấn luyện 5 rounds cho cả 4 mô hình B0, B1, M1, M2, kèm ma trận nhầm lẫn và biểu đồ trực quan.
3. 📊 **`DuongMinhDuy_final.pptx`**: Slide báo cáo thuyết trình với bố cục chuyên nghiệp, trực quan, phục vụ buổi báo cáo cuối kỳ.

---

## 📚 9. Tài liệu Tham khảo

1. **Piczak, K. J. (2015).** *ESC: Dataset for Environmental Sound Classification.* In Proceedings of the 23rd ACM International Conference on Multimedia (pp. 1015–1018).
2. **Piczak, K. J. (2015).** *Environmental sound classification with convolutional neural networks.* In 2015 IEEE 25th International Workshop on Machine Learning for Signal Processing (MLSP) (pp. 1–6).
3. **Park, D. S., et al. (2019).** *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech 2019, pp. 2613–2617.
4. **Bajzik, J., & Jarina, R. (2022).** *Exploiting hierarchy in environmental sound classification.* 32nd International Conference Radioelektronika (RADIOELEKTRONIKA). DOI: 10.1109/RADIOELEKTRONIKA54537.2022.9764900.
5. **Banbury, C., et al. (2020).** *MicroNets: Neural Network Architectures for Deploying TinyML Applications on Commodity Microcontrollers.* arXiv preprint arXiv:2010.11267.
6. **McFee, B., et al. (2024).** *librosa/librosa: 0.11.0.* Zenodo. https://doi.org/10.5281/zenodo.10849767.

---

*Copyright © 2026 Dương Minh Duy. Trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh.*
