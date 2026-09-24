# CLAVE: Cross-Fitted Clause-Level Argument Verification for Scientific Event Extraction

*(CLAVE: Thẩm định luận cứ mức mệnh đề với cross-fitting cho trích xuất sự kiện khoa học)*

**CLAVE** = bộ đề xuất **CARVE-simple** + bộ thẩm định **HONE**. Tên gọi tóm tắt phương pháp: thẩm định (**ve**rification) luận cứ (**a**rgument) ở mức mệnh đề (**cl**ause-level), với bộ thẩm định được huấn luyện trên ứng viên cross-fitted (out-of-fold).

*Bản viết tiếng Việt dạng bài báo. Số liệu chi tiết và nguồn gốc của từng con số nằm ở `docs/PAPER_NOTES.md`.*

---

## Tóm tắt

**Bài toán.** Trích xuất luận cứ trên SciEvent (Dong và cs., EMNLP 2025) thực chất là **phân đoạn span ở mức mệnh đề**: các luận cứ được chấm điểm là những mệnh đề dài, gần như liền kề và không chồng lấn nhau.

**Phương pháp** gồm hai giai đoạn:

1. **CARVE-simple** (bộ đề xuất). DeBERTa-v3-large với **một** đầu BIO duy nhất cho 13 loại span, cộng một đầu dự đoán loại sự kiện. Đây là phiên bản rút gọn của CARVE: hai quyết định thiết kế của bản gốc đã bị ablation có đối chứng chứng minh là **trơ**, nên được bỏ.
2. **HONE** (bộ thẩm định). Một cross-encoder đọc từng span ứng viên trong ngữ cảnh cả cửa sổ (đánh dấu bằng `<a> … </a>`), kèm bằng chứng từ bộ đề xuất. Nó quyết định giữ, loại hay gán lại vai trò cho span đó. HONE được huấn luyện trên **ứng viên out-of-fold**, vì bộ đề xuất học thuộc tập train: 98.1–98.6% ứng viên in-sample là đúng qua ba seed, trong khi out-of-fold chỉ 37.0–38.5%.

**Kết quả chính trên test** (trung bình ± độ lệch chuẩn, ba cặp seed 42/42, 13/13, 101/101; một luật chung đóng băng trên dev). Hàng 1 seed là bản phát hành ban đầu:

| chỉ số | CLAVE 3 seed | CLAVE 1 seed | OneIE | GPT 5-shot |
|---|---|---|---|---|
| Arg-C IoU F1 | **52.45 ± 0.73** | 52.36 | 41.61 | — |
| Arg-I IoU F1 | **60.49 ± 1.25** | 60.09 | 53.57 | — |
| trigger ROUGE-L F1 | **77.22 ± 0.81** | 77.82 | — | 75.08 |

- Cận dưới khoảng tin cậy bootstrap 95% của hệ thống ba seed cao hơn điểm OneIE trên cả hai chỉ số luận cứ (Arg-C 47.87, Arg-I 56.22). Đây là so sánh với điểm tổng hợp được công bố, không phải bootstrap ghép cặp với dự đoán OneIE.
- CLAVE hơn CARVE-simple **+1.72 Arg-C** trên test, CI ghép cặp [+0.24; +3.30]. Trên dev, mức hơn chỉ **+0.33**, dưới ngưỡng +0.5 đăng ký trước cho giả thuyết G1. Kết quả 1 seed trước đây là +1.66 [−0.14; +3.44].

---

## 1. Giới thiệu

### 1.1 Bối cảnh

SciEvent gán cho mỗi đoạn tóm tắt khoa học (một "cửa sổ") đúng một sự kiện:
- một trong 4 loại: Background, Methods, Results, Conclusions;
- 9 vai trò luận cứ được chấm điểm: Context, Method, Results, Challenge, Purpose, Implications, Analysis, Contradictions, Ethical;
- một bộ ba ⟨Agent, Action, Object⟩ dùng để tóm tắt trigger.

Các baseline trong bài gốc gồm EEQA, DEGREE, OneIE và LLM few-shot. Tất cả đều coi luận cứ như **thực thể** cần trích xuất. Nhưng luận cứ ở đây là **mệnh đề**: dài khoảng một nửa cửa sổ, nối đuôi nhau, và gần như không chồng lấn. Vì vậy một bộ gán nhãn span mức từ, đi kèm luật giải mã được hiệu chỉnh theo độ đo, vượt các baseline này với biên độ lớn. Đây là kết quả của CARVE (bản trước của kho mã này).

### 1.2 Quan sát dẫn tới HONE

Toàn bộ mức tăng của CARVE so với baseline thô đến từ **một ngưỡng độ tin cậy**: bỏ những span có xác suất trung bình dưới τ. Trên dev, điều này đưa Arg-C từ 38.88 lên 48.00. Nhưng ngưỡng này là một bộ thẩm định rất thô:
- nó chỉ nhìn vào trung bình xác suất token, không đọc nội dung span;
- nó không thể sửa một vai trò bị gán sai.

Trong phân tích một seed ban đầu, oracle thẩm định trên ứng viên CARVE-simple đạt **72.20** dev Arg-C, so với 49.19 của ngưỡng. Phân tích ba seed dưới luật mới cho các mốc oracle tương ứng **69.91 / 73.47 / 72.44** (§6.3). Khoảng trống đó là động lực cho HONE: thay ngưỡng bằng một mô hình **học** được cách thẩm định.

### 1.3 Đóng góp

1. **CARVE-simple.** Một bộ gán nhãn span gọn hơn CARVE: một đầu BIO gộp, không điều kiện hoá theo loại sự kiện. Kết quả tương đương CARVE trên test (50.73 ± 0.19 so với 50.48 ± 1.15; Δ +0.25, khoảng tin cậy [−1.66, +2.12]), và phương sai giữa các seed thấp hơn khoảng 6 lần.
2. **HONE.** Một bộ thẩm định huấn luyện trên ứng viên out-of-fold. Bỏ cross-fitting giảm **1.63 điểm dev** và **3.48 điểm test** Arg-C trong thí nghiệm ba seed; phép đo phát triển trước đây trên CARVE gốc giảm 4.3 điểm dev.
3. **CLAVE, ba seed:** 52.45 ± 0.73 Arg-C, 60.49 ± 1.25 Arg-I, 77.22 ± 0.81 ROUGE-L trên test. Bản 1 seed đã phát hành đạt 52.36 / 60.09 / 77.82.
4. **Một quy trình báo cáo minh bạch:** mọi lần chạm vào test đều được ghi lại; kiểm toán rò rỉ, kể cả phát hiện về một đặc trưng định danh rò rỉ loại sự kiện (§8); và các giả thuyết bị bác bỏ được báo cáo như kết quả.

---

## 2. Bài toán và cách chấm điểm

**Dữ liệu:** split chính thức gồm 1,278 / 158 / 163 cửa sổ (train / dev / test), mỗi cửa sổ trung bình khoảng 62 từ.

**Chấm điểm:** dùng bộ chấm chính thức `EM_overlap_eval.py`, được import nguyên bản.
- **Arg-I / Arg-C:**
  - so khớp một-đối-một, tham lam, với IoU > 0.5 theo token;
  - loại sự kiện phải khớp;
  - Arg-C còn yêu cầu đúng vai trò;
  - không phụ thuộc trigger;
  - Agent, PrimaryObject và SecondaryObject **không** được chấm.
- **Trigger ROUGE-L:** so chuỗi ⟨Agent, Action, Object⟩ dự đoán với chuỗi vàng.

**Kiểm tra lớp bọc:** đưa nhãn vàng qua bộ xuất dự đoán của chúng tôi rồi vào bộ chấm chính thức cho **100.00** trên mọi chỉ số, cả dev lẫn test. Vậy lớp bọc không làm sai điểm.

---

## 3. Phương pháp

### 3.1 Tổng quan

```
cửa sổ ──► CARVE-simple ──► mọi span argmax thuộc 9 vai trò (không ngưỡng) + 17 đặc trưng
                                      │
                                      ▼
        "event type: <T> . w1 … <a> span </a> … wn"  +  đặc trưng của bộ đề xuất
                                      │
                          HONE: cross-encoder DeBERTa-v3-large
                                      │
                                      ▼
                 {reject, Context, Method, Results, …} cho từng ứng viên
                                      │
                                      ▼
   điểm giữ ≥ θ · vai trò = trộn α · loại sự kiện kết hợp λ · không chồng lấn
```

### 3.2 Giai đoạn 1: CARVE-simple

- **Kiến trúc:**
  - DeBERTa-v3-large, lấy sub-token đầu tiên của mỗi từ làm biểu diễn từ;
  - **một** đầu BIO cho 13 loại span (9 vai trò + Agent, Action, PrimaryObject, SecondaryObject);
  - một đầu phân loại loại sự kiện (4 lớp), dùng attention pooling.
- **Huấn luyện:**
  - loss = CE(BIO) + 0.5·CE(loại);
  - 30 epoch, AdamW (lr encoder 1e-5, lr head 1e-4).
  - Checkpoint được chọn trên dev dưới **giải mã có hiệu chỉnh**, không phải argmax.
- **Vì sao "simple":** CARVE gốc có thêm hai quyết định thiết kế:
  1. hai đầu BIO tách rời cho hai nhóm span, vì hai nhóm chồng lấn nhau ở 5.9% cửa sổ;
  2. điều kiện hoá đầu vai trò bằng loại sự kiện dự đoán, vì độ đo nhạy với loại.

  Ablation đơn yếu tố (3 seed) cho thấy cả hai đều **trơ**: bỏ đầu tách rời +0.75 ± 1.6, bỏ điều kiện hoá −0.24 ± 1.6. Bỏ cả hai cùng lúc cho 48.50 ± 0.68 so với 47.58 ± 0.55 trên cùng lưới. **Phép đo thống kê dữ liệu là đúng, nhưng suy luận kiến trúc rút ra từ nó thì không.** Đầu dự đoán loại sự kiện vẫn cần, vì hệ thống phải phát ra một loại.
- **Ghi chú kỹ thuật:** transformers ≥ 5 nạp DeBERTa-v3-large ở fp16, khiến AdamW cho NaN ngay bước đầu. Vì vậy trọng số chính phải để fp32, còn bf16 chỉ dùng qua autocast.

### 3.3 Ứng viên và cross-fitting

**Ứng viên:**
- là mọi span argmax thuộc 9 vai trò, **không** lọc ngưỡng;
- mỗi ứng viên mang 17 đặc trưng: khối vai trò theo từng vai trò, P(O), độ tin cậy trung bình và lớn nhất, độ dài, và vị trí.

**Vấn đề:** nếu lấy ứng viên mà bộ đề xuất sinh ra trên chính tập train của nó, thì 98.1–98.6% là đúng qua ba seed. Phân bố nhãn này không đại diện cho ứng viên ngoài mẫu.

**Cross-fitting:**
- chia train thành 5 fold theo cửa sổ, phân tầng theo loại sự kiện;
- với mỗi fold, huấn luyện lại CARVE-simple trên 4 fold còn lại, rồi giải mã fold bị giữ lại;
- nhờ đó mỗi cửa sổ train nhận ứng viên từ một mô hình chưa từng thấy nó, đúng như quan hệ của dev/test với mô hình huấn luyện trên toàn bộ train.

| ứng viên (seed 42) | số lượng | mỗi cửa sổ | tỉ lệ đúng |
|---|---|---|---|
| train, out-of-fold | 6,468 | 5.06 | 38.0% |
| train, in-sample (đối chứng) | 3,897 | 3.05 | 98.46% |
| dev | 792 | 5.01 | 42.2% |

Thống kê đủ ba seed nằm trong `experiment/tables/T8.md`.

### 3.4 Giai đoạn 2: bộ thẩm định HONE

- **Đầu vào:** `event type: <loại dự đoán> .` + cửa sổ, trong đó ứng viên được bọc bởi hai token đặc biệt `<a> … </a>` (kỹ thuật entity marker).
- **Biểu diễn:** [h_CLS ; h_<a> ; h_</a> ; MLP(17 đặc trưng)] → MLP → 10 lớp {reject, 9 vai trò}.
- **Nhãn** theo đúng độ đo: vai trò của span vàng có IoU > 0.5 với ứng viên, ngược lại là `reject`.
- **Huấn luyện:** 5 epoch, batch 8 × 2; epoch được chọn trên dev.

### 3.5 Giai đoạn 3: giải mã

- **Điểm giữ** = 1 − p(reject).
- **Vai trò** = trộn giữa phân phối vai trò của verifier (trọng số α) và khối vai trò của bộ đề xuất (1 − α).
  Lý do (đo trong quá trình phát triển): verifier **lọc** span tốt hơn (AUC 0.890 so với 0.860), nhưng **gán vai trò** kém hơn bộ đề xuất (77.1% so với 81.3%).
- **Loại sự kiện** = argmax_t [log p(t) + λ·Σ log P(vai trò | t)], với P(vai trò | t) ước lượng từ nhãn vàng của train.
- **Chọn span:** tham lam theo điểm giữ ≥ θ, bỏ ứng viên chồng lấn với span đã nhận.
- **Luật ba seed đóng băng trên dev:** θ = 0.30, min_len = 3, luật chồng lấn `any`, α = 0.50, λ = 0.5 (`experiment/assets/decoding_rule_3seed.json`). Bản 1 seed cũ dùng θ = 0.20, min_len = 1, α = 0.25, λ = 0.5.

---

## 4. Thiết lập thí nghiệm

**Giao thức:**
- mọi lựa chọn (checkpoint, epoch, tham số giải mã) đều làm trên dev;
- luật của từng cấu hình được đóng băng trước lần đánh giá test đã đăng ký của cấu hình đó;
- luật ba seed của T-main được ghi lúc 13:45:39 ngày 2026-09-24, trước dòng test-log và ba file dự đoán. Mốc 19:31:30 → 19:31:40 là của bản một seed phát hành trước đó.

**Lịch sử chạm vào test** (công bố đầy đủ ở PAPER_NOTES §3):
- Trong quá trình phát triển, nhiều cấu hình khác đã được đánh giá trên test: CARVE gốc, HONE đa seed, HONE 1 seed với bộ đề xuất gốc, và CARVE-simple. Mỗi lần đều đóng băng luật trên dev trước.
- Cấu hình của **bản một seed phát hành trước** được chọn sau khi đã thấy các kết quả đó. Nó không phải cấu hình có điểm test cao nhất, nhưng đó vẫn là lựa chọn hậu nghiệm.
- Thí nghiệm ba seed mới đăng ký trước T-main, T-A1–A7, T-P1 và T-P2 tuỳ chọn trong `experiment/EXPERIMENT_PLAN.md`; mọi lần chạm test nằm ở `experiment/TEST_LOG.md`. Pha 2 có một chẩn đoán hậu nghiệm lỡ xem nhãn test của ứng viên seed 13; đã ghi rõ trong log, không dùng để chọn mô hình hay luật.

**Seed:** kết quả chính mới dùng ba cặp *(bộ đề xuất, verifier)* (42,42), (13,13), (101,101), cùng một luật chọn bằng trung bình dev; bản phát hành ban đầu dùng một cặp (42,42).

---

## 5. Kết quả

### 5.1 Trigger ROUGE-L (so với Bảng 3 của bài gốc)

| Phương pháp | P | R | **F1** |
|---|---|---|---|
| EEQA | 81.93 | 34.57 | 45.05 |
| DEGREE | 64.56 | 63.49 | 56.85 |
| OneIE | 73.73 | 79.40 | 72.40 |
| GPT (5-shot) — tốt nhất bài gốc | 73.70 | 78.82 | 75.08 |
| Qwen (2-shot) | 57.27 | 69.71 | 61.18 |
| Llama (0-shot) | 54.88 | 61.07 | 55.83 |
| CARVE-simple (3 seed) | 84.77 ± 0.43 | 76.15 ± 1.21 | 77.22 ± 0.81 |
| CLAVE 1 seed (bản phát hành cũ) | 85.13 | 76.78 | 77.82 |
| **CLAVE 3 seed** | **84.77 ± 0.43** | **76.15 ± 1.21** | **77.22 ± 0.81** |

ROUGE-L của hệ thống **bằng đúng** của bộ đề xuất: trigger và bộ ba AAO lấy thẳng từ CARVE-simple, không đi qua HONE. Chúng tôi không có dự đoán từng cửa sổ của GPT, nên không kiểm định được ý nghĩa của mức +2.14 điểm trung bình so với GPT 5-shot (bản một seed cũ là +2.74).

### 5.2 Trích xuất luận cứ, IoU > 0.5 (so với Bảng 4 của bài gốc)

| Phương pháp | ArgI-P | ArgI-R | **ArgI-F1** | ArgC-P | ArgC-R | **ArgC-F1** |
|---|---|---|---|---|---|---|
| EEQA | 32.09 | 33.77 | 32.91 | 25.85 | 27.20 | 26.51 |
| DEGREE | 67.79 | 19.13 | 29.84 | 48.99 | 13.83 | 21.57 |
| OneIE — tốt nhất bài gốc | 51.11 | 56.29 | 53.57 | 39.69 | 43.71 | 41.61 |
| GPT (5-shot) | 50.04 | 49.93 | 49.98 | 34.51 | 34.42 | 34.47 |
| CARVE-simple (3 seed) | 62.64 ± 0.23 | 55.03 ± 1.08 | 58.58 ± 0.57 | 54.24 ± 0.46 | 47.65 ± 0.65 | 50.73 ± 0.19 |
| CARVE-simple (seed 42) | 62.91 | 54.41 | 58.35 | 54.66 | 47.28 | 50.70 |
| CLAVE 1 seed (bản phát hành cũ) | 70.18 | 52.53 | 60.09 | 61.15 | 45.78 | 52.36 |
| **CLAVE 3 seed** | **69.93 ± 6.63** | **53.97 ± 6.23** | **60.49 ± 1.25** | **60.77 ± 7.27** | **46.72 ± 4.01** | **52.45 ± 0.73** |
| *Δ điểm trung bình so với OneIE* | *+18.82* | *−2.32* | *+6.92* | *+21.08* | *+3.01* | *+10.84* |

### 5.3 Ý nghĩa thống kê

Bootstrap ghép cặp: 5,000 lần lấy mẫu lại 163 cửa sổ test, hai hệ thống dùng cùng tập cửa sổ trong mỗi lần.

| so sánh | Arg-C Δ [95% CI] | Arg-I Δ [95% CI] |
|---|---|---|
| CLAVE 3 seed so với CARVE-simple | **+1.72 [+0.24, +3.30]** | **+1.90 [+0.21, +3.72]** |
| CLAVE 3 seed so với CARVE gốc | +1.97 [−0.20, +4.07] | +2.54 [+0.45, +4.65] |
| CLAVE 1 seed so với CARVE-simple cùng seed (archive) | +1.66 [−0.14, +3.44] | +1.74 [−0.65, +4.03] |
| CLAVE 1 seed so với CARVE gốc seed 42 (archive) | +3.14 [−0.62, +6.83] | +4.84 [+0.86, +8.83] |
| CARVE-simple so với CARVE gốc (3 seed mỗi bên) | +0.25 [−1.66, +2.12] | +0.64 [−1.27, +2.46] |

Khoảng tin cậy bootstrap 95% tuyệt đối của CLAVE ba seed: Arg-C [47.87, 57.08], Arg-I [56.22, 64.61]. Cả hai cận dưới đều trên điểm OneIE được công bố; không có dự đoán OneIE từng cửa sổ để kiểm định ghép cặp.

### 5.4 Bốn chế độ so khớp (test)

| chế độ | ArgI-F1 | ArgC-F1 |
|---|---|---|
| Exact Match | 39.62 ± 1.99 | 36.74 ± 2.77 |
| Simple overlap | 70.33 ± 2.03 | 59.96 ± 0.78 |
| SciREX > 0.5 | 65.92 ± 1.91 | 56.21 ± 0.76 |
| **IoU > 0.5** | **60.49 ± 1.25** | **52.45 ± 0.73** |

Precision và recall của từng chế độ có trong `experiment/tables/T3.md`; PAPER_NOTES §4.3 giữ số của bản một seed.

### 5.5 Dev

| hệ thống | dev Arg-C |
|---|---|
| CARVE gốc (3 seed) | 47.14 ± 0.26 |
| CARVE-simple (3 seed) | 48.10 ± 0.96 |
| CARVE-simple (seed 42) | 49.19 |
| CLAVE 1 seed, luật cũ | 49.16 |
| **CLAVE 3 seed, luật chung** | **48.43 ± 0.47** |

---

## 6. Phân tích

### 6.1 HONE thay đổi điều gì

So với bộ đề xuất cùng seed, HONE:
- **tăng precision:** Arg-C P +6.52, Arg-I P +7.30;
- **giảm recall:** Arg-C R −0.94, Arg-I R −1.06.

Các mức dịch chuyển là trung bình ba seed; bảng theo vai trò, loại sự kiện, lĩnh vực, độ dài và nhóm lỗi ở `experiment/tables/T7.md`. Bản 1 seed trước đây dự đoán 399 luận cứ so với 533 luận cứ vàng. Trên test ba seed, bộ thẩm định tăng precision nhưng mức hơn trên dev không vượt ngưỡng +0.5 của G1.

### 6.2 Vì sao phần cộng thêm nhỏ khi dùng CARVE-simple

Cùng một thiết kế HONE (1 seed), nhưng phần cộng thêm phụ thuộc vào bộ đề xuất:

| bộ đề xuất (seed 42) | ngưỡng (test) | + HONE (test) | phần cộng thêm |
|---|---|---|---|
| CARVE gốc | 49.22 | 53.35 | **+4.13** [+1.57, +6.74] |
| CARVE-simple | 50.70 | 52.36 | +1.66 [−0.14, +3.44] |

- Đây là so sánh của bản một seed trong quá trình phát triển. Hai hệ thống sau HONE có Δ −0.99, CI [−4.41; +2.17]; trên dev bản một seed, 49.16 so với 49.19.
- Trong thí nghiệm ba seed mới, CLAVE đạt 48.43 so với CARVE-simple 48.10 trên dev, thấp hơn ngưỡng +0.5 đăng ký trước. Trên test, mức hơn là +1.72 với CI [+0.24; +3.30]. Hai kết luận này được giữ riêng.

### 6.3 Còn bao nhiêu khoảng trống

Với luật chung ba seed, oracle giữ span và gán vai trò đạt **69.91 / 73.47 / 72.44** Arg-C trên dev, so với **48.97 / 48.17 / 48.14** thực tế. Oracle chỉ sửa loại sự kiện **ở bước giải mã** đạt **52.86 / 52.12 / 51.86**, không chạy lại verifier với loại vàng. Oracle sửa cả ba đạt **78.93 / 80.67 / 80.39**. Mốc 72.20 trước đây là seed 42 dưới luật một seed khác, nên không đối chiếu trực tiếp với hàng 69.91. Bảng chi tiết ở `experiment/tables/T7.md`.

---

## 7. Ablation ba seed và bằng chứng phát triển

Mọi hàng CLAVE dưới đây dùng ba cặp seed và luật riêng chọn trên dev. Δ test là biến thể trừ CLAVE đầy đủ; CI bootstrap ghép cặp 5.000 mẫu. T5 chứa thêm Arg-I, P/R và tên file luật.

| biến thể | Dev Arg-C F1 | Test Arg-C F1 | Δ test [95% CI] |
|---|---:|---:|---:|
| CLAVE đầy đủ | 48.43 ± 0.47 | **52.45 ± 0.73** | 0 |
| bỏ verifier (CARVE-simple) | 48.10 ± 0.96 | 50.73 ± 0.19 | −1.72 [−3.30; −0.24] |
| bỏ cross-fitting | 46.80 ± 0.79 | 48.97 ± 0.61 | −3.48 [−4.92; −2.10] |
| bỏ đặc trưng proposer | 49.08 ± 1.45 | 51.46 ± 1.22 | −0.99 [−2.03; +0.00] |
| chỉ vai trò verifier (α=1) | 47.36 ± 1.08 | 50.00 ± 1.58 | −2.46 [−4.35; −0.56] |
| chỉ vai trò proposer (α=0) | 48.06 ± 0.59 | 52.16 ± 0.31 | −0.29 [−0.73; +0.08] |
| bỏ kết hợp loại sự kiện (λ=0) | 48.35 ± 0.34 | 52.30 ± 0.82 | −0.15 [−0.48; +0.00] |
| verifier thuần (α=1, λ=0) | 47.14 ± 1.07 | 49.62 ± 1.23 | −2.83 [−4.64; −1.00] |
| NMS iou / wis | 48.43 ± 0.47 | 52.45 ± 0.73 | 0 |

Theo tiêu chí đăng ký trước trên dev (+0.5 Arg-C), **G1 bị bác bỏ**, **G2 được ủng hộ**, **G3 bị bác bỏ**, **G4 được ủng hộ**, **G5 bị bác bỏ**. Test được báo cáo tách biệt, không dùng để đổi quyết định này. Tài liệu nguồn: `experiment/EXPERIMENT_PLAN.md`, `experiment/tables/T5.md` và các báo cáo Pha 4–6.

**Lưu ý về P/R.** Dưới luật chung (θ 0.30), ba run đạt F1 gần nhau nhưng ở các điểm vận hành khác nhau:

| cặp seed | Arg-C P / R / F1 |
|---|---|
| 42/42 | 64.36 / 45.40 / 53.25 |
| 13/13 | 52.40 / 51.22 / 51.80 |
| 101/101 | 65.54 / 43.53 / 52.31 |

Verifier seed 13 được chọn ở epoch 1 nên giữ nhiều span hơn. Vì vậy độ lệch chuẩn của P/R (±7.27 / ±4.01) lớn hơn nhiều so với của F1 (±0.73).

### 7.1 Thiết kế bộ đề xuất: hiệu chỉnh và bảng 2×2 trên test (T-P1, T-P2)

Mỗi hàng có một luật chung đóng băng trên dev (trung bình 3 seed) trước khi test. Δ là so với CARVE-simple, bootstrap ghép cặp.

| bộ đề xuất (3 seed) | Dev Arg-C | Test Arg-I F1 | Test Arg-C F1 | Δ Arg-C [95% CI] |
|---|---:|---:|---:|---:|
| CARVE-simple, giải mã argmax (T-P1) | 38.32 ± 0.25 | 47.87 ± 1.31 | 39.67 ± 0.76 | −11.07 [−13.36; −8.85] |
| 2 đầu + điều kiện hoá (CARVE gốc) | 47.14 ± 0.26 | 57.95 ± 2.46 | 50.48 ± 1.15 | −0.25 [−2.12; +1.66] |
| 2 đầu, không điều kiện hoá (T-P2a) | 47.13 ± 1.32 | 57.28 ± 0.73 | 49.82 ± 0.62 | −0.91 [−2.73; +0.97] |
| 1 đầu + điều kiện hoá (T-P2b) | 48.59 ± 3.13 | 58.48 ± 0.56 | 50.76 ± 0.39 | +0.03 [−1.41; +1.51] |
| **1 đầu, không điều kiện hoá (CARVE-simple)** | 48.10 ± 0.96 | 58.58 ± 0.57 | 50.73 ± 0.19 | — |

- **Hiệu chỉnh giải mã** là thành phần duy nhất của bộ đề xuất tạo khác biệt lớn: khoảng +11 điểm test.
- **Trong bảng 2×2 trên test, không ô nào khác CARVE-simple.** Điều này xác nhận trên test rằng tách hai đầu và điều kiện hoá đều trơ.
- Hai ô T-P2 được huấn luyện lại bằng code archive (giống hệt từng byte), trong bản sao cô lập `experiment/archive_carve_full/`.
- Lượt test đầu của T-P2a bị crash do lỗi thư mục ra của bộ decode archive. Nó được chạy lại với cùng luật và cùng checkpoint, và đã ghi trong TEST_LOG.

### 7.2 Bằng chứng lịch sử

Mỗi dòng dưới đây ghi rõ cấu hình của quá trình phát triển trước thí nghiệm ba seed. "Gốc" nghĩa là bộ đề xuất CARVE gốc (2 đầu), seed 42.

| quyết định | bằng chứng (dev Arg-C) | cấu hình |
|---|---|---|
| ngưỡng hiệu chỉnh thay vì argmax | 38.88 → 48.00 | CARVE gốc |
| một đầu gộp thay vì hai đầu | +0.75 ± 1.6 (trơ) | CARVE gốc, 3 seed |
| không điều kiện hoá loại sự kiện | −0.24 ± 1.6 (trơ) | CARVE gốc, 3 seed |
| không CRF / không head vai trò mức span / không LLRD | −0.28 / −0.96 / −2.20 | CARVE gốc, 3 seed |
| cross-fitting (out-of-fold) | 49.16 so với 44.87 khi huấn luyện in-sample | HONE, gốc |
| đặc trưng của bộ đề xuất trong verifier | 49.16 so với 48.24 chỉ dùng văn bản | HONE, gốc |
| ghép các ứng viên liền kề | 48.82 so với 49.16: **bác bỏ** | HONE, gốc |
| ensemble nhiều verifier | +0.10: **bác bỏ** | HONE, gốc, gộp seed |

---

## 8. Kiểm toán

- **Không trùng lặp dữ liệu:** số cửa sổ chung giữa train/dev/test là 0. Bộ chấm chính thức được import nguyên bản; oracle đạt 100.00.
- **Phép chia chính thức là theo cửa sổ, không theo tài liệu.** 143/147 tóm tắt trong test có đoạn khác nằm trong train. Mọi baseline đã công bố đều chịu cùng tính chất này, nên so sánh vẫn ngang hàng; nhưng điều này phải được nêu rõ.
- **Bẫy rò rỉ `sent_id`.** Hậu tố của mã cửa sổ (`-0`, `-1`, …) đoán đúng loại sự kiện vàng 94.9% (train), 98.7% (dev), 93.9% (test), vì nó là thứ tự của sự kiện trong annotation. Không mô hình nào của chúng tôi đọc nó. Một hệ thống vô tình dùng nó sẽ đạt độ chính xác loại sự kiện gần như tuyệt đối mà không giải bài toán.
- **Dự đoán test của bản một seed:**
  - đủ 163/163 cửa sổ;
  - 399 luận cứ so với 533 vàng, tức dự đoán thiếu, nên không có chuyện ăn gian precision bằng số lượng;
  - 0 cặp (span, vai trò) trùng lặp.
- **Không dùng loại sự kiện vàng khi suy luận.** Con số độ chính xác loại sự kiện 88.96% là của bản một seed phát hành trước đó.

---

## 9. Hạn chế

1. **Chỉ ba seed.** Độ lệch chuẩn Arg-C test là 0.73; G1 vẫn bị bác bỏ theo ngưỡng dev đã đăng ký dù CI ghép cặp trên test dương. Bản phát hành ban đầu chỉ có một seed.
2. **Lựa chọn hậu nghiệm.** Cấu hình cuối được chọn sau nhiều lần đánh giá test; điều này được công bố đầy đủ.
3. **Không phải cấu hình mạnh nhất đã đo.** HONE với bộ đề xuất gốc đạt 53.35 (1 seed), và HONE gộp 3 seed đạt 53.06 ± 1.04 Arg-C / 63.05 Arg-I. Các cấu hình này được lưu trữ trong kho SciEvent.
4. **Một số ablation phát triển dùng CARVE gốc.** Các ablation verifier/cross-fitting/đặc trưng chính đã được lặp lại trên CARVE-simple ở ba seed (Bảng T5); những biến thể khác của quá trình phát triển vẫn là số archive.
5. **Tái lập:** suy luận từ checkpoint tái lập chính xác từng byte, nhưng huấn luyện lại trên GPU không tất định. Cần kỳ vọng dao động cỡ một seed.
6. **Chồng lấn tài liệu trong split chính thức** (§8).

---

## 10. Kết luận

**Về bài toán:** trích xuất luận cứ trên SciEvent là phân đoạn span mức mệnh đề. Một bộ gán nhãn span đơn giản, kèm giải mã có hiệu chỉnh, đã vượt baseline tốt nhất của bài gốc khoảng 9 điểm Arg-C.

**Về HONE:** thêm bộ thẩm định học từ ứng viên out-of-fold đưa CLAVE ba seed lên 52.45 ± 0.73 Arg-C, hơn điểm OneIE 10.84. Phần hơn CARVE-simple trên test là +1.72 [+0.24; +3.30], còn chênh dev +0.33 không đạt ngưỡng +0.5 đăng ký trước. Bỏ cross-fitting giảm 1.63 điểm dev và 3.48 điểm test.

**Hai bài học về phương pháp luận:**
1. Suy luận kiến trúc từ thống kê dữ liệu cần được ablation kiểm chứng. Ở đây, hai quyết định như vậy của CARVE gốc đều không đứng vững.
2. Một bộ thẩm định chỉ học được điều gì đó khi dữ liệu huấn luyện của nó có lỗi thật. Cross-fitting là điều kiện cần.

**Hướng tiếp theo:**
- mở rộng quá ba seed và đánh giá trên phép chia theo tài liệu;
- nhắm vào khoảng trống giữa mốc oracle giữ span/vai trò và điểm thực tế trên dev.

---

## Phụ lục A. Tái lập

```bash
pip install -e .
bash scripts/setup_data.sh
python3 tests/test_contract.py
bash scripts/reproduce.sh 3-seed 1
# đọc báo cáo STOP của từng pha, rồi chạy 2, 3, …, 7
# nếu T-P2 đã được duyệt: CLAVE_RUN_P2=1 bash scripts/reproduce.sh 3-seed 7
```

Chi tiết từng bước, tham số và các kiểm tra tương đương đã thực hiện: README, PAPER_NOTES §8/§10 và `experiment/EXPERIMENT_PLAN.md`. Bản phát hành một seed cũ dùng `bash scripts/reproduce.sh released`.
