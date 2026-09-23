# CLAVE: Cross-Fitted Clause-Level Argument Verification for Scientific Event Extraction

*(CLAVE: Thẩm định luận cứ mức mệnh đề với cross-fitting cho trích xuất sự kiện khoa học)*

**CLAVE** = bộ đề xuất **CARVE-simple** + bộ thẩm định **HONE**. Tên gọi tóm tắt phương pháp: thẩm định (**ve**rification) luận cứ (**a**rgument) ở mức mệnh đề (**cl**ause-level), với bộ thẩm định được huấn luyện trên ứng viên cross-fitted (out-of-fold).

*Bản viết tiếng Việt dạng bài báo. Số liệu chi tiết và nguồn gốc của từng con số nằm ở `docs/PAPER_NOTES.md`.*

---

## Tóm tắt

**Bài toán.** Trích xuất luận cứ trên SciEvent (Dong và cs., EMNLP 2025) thực chất là **phân đoạn span ở mức mệnh đề**: các luận cứ được chấm điểm là những mệnh đề dài, gần như liền kề và không chồng lấn nhau.

**Phương pháp** gồm hai giai đoạn:

1. **CARVE-simple** (bộ đề xuất). DeBERTa-v3-large với **một** đầu BIO duy nhất cho 13 loại span, cộng một đầu dự đoán loại sự kiện. Đây là phiên bản rút gọn của CARVE: hai quyết định thiết kế của bản gốc đã bị ablation có đối chứng chứng minh là **trơ**, nên được bỏ.
2. **HONE** (bộ thẩm định). Một cross-encoder đọc từng span ứng viên trong ngữ cảnh cả cửa sổ (đánh dấu bằng `<a> … </a>`), kèm bằng chứng từ bộ đề xuất. Nó quyết định giữ, loại hay gán lại vai trò cho span đó. HONE được huấn luyện trên **ứng viên out-of-fold**, vì bộ đề xuất học thuộc tập train: 97.5% ứng viên in-sample là đúng, trong khi ngoài mẫu chỉ khoảng 38%.

**Kết quả trên test** (1 seed, luật giải mã đóng băng trên dev):

| chỉ số | CLAVE | OneIE (tốt nhất về luận cứ trong bài gốc) | GPT 5-shot (tốt nhất về trigger) |
|---|---|---|---|
| Arg-C IoU F1 | **52.36** | 41.61 | — |
| Arg-I IoU F1 | **60.09** | 53.57 | — |
| trigger ROUGE-L F1 | **77.82** | — | 75.08 |

- Cận dưới khoảng tin cậy bootstrap 95% của hệ thống vẫn cao hơn OneIE trên cả hai chỉ số luận cứ (Arg-C 47.28, Arg-I 55.12).
- Phần cộng thêm của HONE so với chính bộ đề xuất của nó là +1.66 Arg-C, khoảng tin cậy [−0.14, +3.44]. **Chưa có ý nghĩa thống kê** với 1 seed, và gần bằng 0 trên dev. Chúng tôi nêu điều này ngay từ đầu.

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

Trong khi đó, nếu có một bộ thẩm định hoàn hảo chạy trên chính các ứng viên của CARVE-simple (qua cùng bộ giải mã), dev Arg-C đạt **72.20**, so với 49.19 của ngưỡng. Khoảng trống đó là động lực cho HONE: thay ngưỡng bằng một mô hình **học** được cách thẩm định.

### 1.3 Đóng góp

1. **CARVE-simple.** Một bộ gán nhãn span gọn hơn CARVE: một đầu BIO gộp, không điều kiện hoá theo loại sự kiện. Kết quả tương đương CARVE trên test (50.73 ± 0.19 so với 50.48 ± 1.15; Δ +0.25, khoảng tin cậy [−1.66, +2.12]), và phương sai giữa các seed thấp hơn khoảng 6 lần.
2. **HONE.** Một bộ thẩm định huấn luyện trên ứng viên out-of-fold. Chúng tôi cho thấy cross-fitting là **bắt buộc**: bỏ đi thì mất 4.3 điểm dev (đo với bộ đề xuất CARVE gốc).
3. **Hệ thống cuối, CLAVE** (1 seed): 52.36 Arg-C, 60.09 Arg-I, 77.82 ROUGE-L trên test.
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

**Vấn đề:** nếu lấy ứng viên mà bộ đề xuất sinh ra trên chính tập train của nó, thì 97.5% là đúng. Một bộ thẩm định học trên dữ liệu đó chỉ học được "giữ hết".

**Cross-fitting:**
- chia train thành 5 fold theo cửa sổ, phân tầng theo loại sự kiện;
- với mỗi fold, huấn luyện lại CARVE-simple trên 4 fold còn lại, rồi giải mã fold bị giữ lại;
- nhờ đó mỗi cửa sổ train nhận ứng viên từ một mô hình chưa từng thấy nó, đúng như quan hệ của dev/test với mô hình huấn luyện trên toàn bộ train.

| ứng viên (seed 42) | số lượng | mỗi cửa sổ | tỉ lệ đúng |
|---|---|---|---|
| train, out-of-fold | 6,468 | 5.06 | 38.0% |
| dev | 792 | 5.01 | 42.2% |

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
- **Luật đóng băng trên dev:** θ = 0.20, min_len = 1, luật chồng lấn `any`, α = 0.25, λ = 0.5.

---

## 4. Thiết lập thí nghiệm

**Giao thức:**
- mọi lựa chọn (checkpoint, epoch, tham số giải mã) đều làm trên dev;
- luật được đóng băng trước khi chạy test, rồi test được chạy **một lần**;
- luật của hệ thống cuối được ghi lúc 19:31:30, và file dự đoán test lúc 19:31:40.

**Lịch sử chạm vào test** (công bố đầy đủ ở PAPER_NOTES §3):
- Trong quá trình phát triển, nhiều cấu hình khác đã được đánh giá trên test: CARVE gốc, HONE đa seed, HONE 1 seed với bộ đề xuất gốc, và CARVE-simple. Mỗi lần đều đóng băng luật trên dev trước.
- Cấu hình cuối được **chọn sau khi** đã thấy các kết quả đó. Nó không phải cấu hình có điểm test cao nhất, nên lựa chọn này không thổi phồng con số báo cáo; nhưng đó vẫn là lựa chọn hậu nghiệm.

**Seed:** hệ thống cuối dùng 1 seed (bộ đề xuất 42, verifier 42). Bộ đề xuất CARVE-simple được đo trên 3 seed.

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
| **CLAVE** | **85.13** | 76.78 | **77.82** |

ROUGE-L của hệ thống **bằng đúng** của bộ đề xuất: trigger và bộ ba AAO lấy thẳng từ CARVE-simple, không đi qua HONE. Chúng tôi không có dự đoán từng cửa sổ của GPT, nên không kiểm định được ý nghĩa của mức +2.74 so với GPT 5-shot.

### 5.2 Trích xuất luận cứ, IoU > 0.5 (so với Bảng 4 của bài gốc)

| Phương pháp | ArgI-P | ArgI-R | **ArgI-F1** | ArgC-P | ArgC-R | **ArgC-F1** |
|---|---|---|---|---|---|---|
| EEQA | 32.09 | 33.77 | 32.91 | 25.85 | 27.20 | 26.51 |
| DEGREE | 67.79 | 19.13 | 29.84 | 48.99 | 13.83 | 21.57 |
| OneIE — tốt nhất bài gốc | 51.11 | 56.29 | 53.57 | 39.69 | 43.71 | 41.61 |
| GPT (5-shot) | 50.04 | 49.93 | 49.98 | 34.51 | 34.42 | 34.47 |
| CARVE-simple (3 seed) | 62.64 ± 0.23 | 55.03 ± 1.08 | 58.58 ± 0.57 | 54.24 ± 0.46 | 47.65 ± 0.65 | 50.73 ± 0.19 |
| CARVE-simple (seed 42) | 62.91 | 54.41 | 58.35 | 54.66 | 47.28 | 50.70 |
| **CLAVE** | **70.18** | 52.53 | **60.09** | **61.15** | 45.78 | **52.36** |
| *Δ so với OneIE* | *+19.07* | *−3.76* | *+6.52* | *+21.46* | *+2.07* | *+10.75* |

### 5.3 Ý nghĩa thống kê

Bootstrap ghép cặp: 5,000 lần lấy mẫu lại 163 cửa sổ test, hai hệ thống dùng cùng tập cửa sổ trong mỗi lần.

| so sánh | Arg-C Δ [95% CI] | Arg-I Δ [95% CI] |
|---|---|---|
| CLAVE so với CARVE-simple (cùng seed) | +1.66 [−0.14, +3.44] | +1.74 [−0.65, +4.03] |
| CLAVE so với CARVE gốc (seed 42) | +3.14 [−0.62, +6.83] | **+4.84 [+0.86, +8.83]** |
| CARVE-simple so với CARVE gốc (3 seed mỗi bên) | +0.25 [−1.66, +2.12] | +0.64 [−1.27, +2.46] |

Khoảng tin cậy 95% tuyệt đối của hệ thống: Arg-C [47.28, 57.18], Arg-I [55.12, 64.72]. Cả hai cận dưới đều trên OneIE.

### 5.4 Bốn chế độ so khớp (test)

| chế độ | ArgI-F1 | ArgC-F1 |
|---|---|---|
| Exact Match | 39.70 | 36.48 |
| Simple overlap | 69.74 | 59.44 |
| SciREX > 0.5 | 65.24 | 56.01 |
| **IoU > 0.5** | 60.09 | 52.36 |

Precision và recall của từng chế độ có trong PAPER_NOTES §4.3.

### 5.5 Dev

| hệ thống | dev Arg-C |
|---|---|
| CARVE gốc (3 seed) | 47.14 ± 0.26 |
| CARVE-simple (3 seed) | 48.10 ± 0.96 |
| CARVE-simple (seed 42) | 49.19 |
| **CLAVE** | **49.16** |

---

## 6. Phân tích

### 6.1 HONE thay đổi điều gì

So với bộ đề xuất cùng seed, HONE:
- **tăng precision mạnh:** Arg-C P +6.49, Arg-I P +7.27;
- **đổi lại mất ít recall:** Arg-C R −1.50.

Hệ thống dự đoán 399 luận cứ so với 533 luận cứ vàng. Nó là một **bộ lọc** span tốt hơn ngưỡng độ tin cậy, chứ không phải bộ gán vai trò tốt hơn; điều này khớp với chẩn đoán trong quá trình phát triển (§3.5).

### 6.2 Vì sao phần cộng thêm nhỏ khi dùng CARVE-simple

Cùng một thiết kế HONE (1 seed), nhưng phần cộng thêm phụ thuộc vào bộ đề xuất:

| bộ đề xuất (seed 42) | ngưỡng (test) | + HONE (test) | phần cộng thêm |
|---|---|---|---|
| CARVE gốc | 49.22 | 53.35 | **+4.13** [+1.57, +6.74] |
| CARVE-simple | 50.70 | 52.36 | +1.66 [−0.14, +3.44] |

- Bộ đề xuất tốt hơn thì còn ít lỗi hơn để bộ thẩm định sửa. Hai hệ thống sau khi qua HONE về cùng một mức: Δ −0.99, khoảng tin cậy [−4.41, +2.17].
- Trên dev, HONE với CARVE-simple không cộng thêm gì (49.16 so với 49.19).
- Kết luận trung thực: **với bộ đề xuất này, lợi ích của bộ thẩm định chưa được xác lập.**

### 6.3 Còn bao nhiêu khoảng trống

Một bộ thẩm định hoàn hảo trên chính các ứng viên dev đạt 72.20 Arg-C, so với 49.16 hiện tại. Phần lớn khoảng trống nằm ở **quyết định giữ/loại**, không phải ở ranh giới span. Đây là hướng cải tiến rõ nhất.

---

## 7. Bằng chứng ablation từ quá trình phát triển

Mỗi dòng ghi rõ cấu hình được dùng để đo. "Gốc" nghĩa là bộ đề xuất CARVE gốc (2 đầu), seed 42. Các ablation của HONE **chưa** được chạy lại trên CARVE-simple.

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
- **Dự đoán test của hệ thống:**
  - đủ 163/163 cửa sổ;
  - 399 luận cứ so với 533 vàng, tức dự đoán thiếu, nên không có chuyện ăn gian precision bằng số lượng;
  - 0 cặp (span, vai trò) trùng lặp.
- **Không dùng loại sự kiện vàng khi suy luận.** Độ chính xác loại sự kiện dự đoán trên test là 88.96%.

---

## 9. Hạn chế

1. **Một seed.** Chưa đo được phương sai giữa các seed của hệ thống cuối. Phần cộng thêm so với bộ đề xuất chưa có ý nghĩa thống kê.
2. **Lựa chọn hậu nghiệm.** Cấu hình cuối được chọn sau nhiều lần đánh giá test; điều này được công bố đầy đủ.
3. **Không phải cấu hình mạnh nhất đã đo.** HONE với bộ đề xuất gốc đạt 53.35 (1 seed), và HONE gộp 3 seed đạt 53.06 ± 1.04 Arg-C / 63.05 Arg-I. Các cấu hình này được lưu trữ trong kho SciEvent.
4. **Ablation của HONE được đo với bộ đề xuất gốc, chưa lặp lại trên CARVE-simple.**
5. **Tái lập:** suy luận từ checkpoint tái lập chính xác từng byte, nhưng huấn luyện lại trên GPU không tất định. Cần kỳ vọng dao động cỡ một seed.
6. **Chồng lấn tài liệu trong split chính thức** (§8).

---

## 10. Kết luận

**Về bài toán:** trích xuất luận cứ trên SciEvent là phân đoạn span mức mệnh đề. Một bộ gán nhãn span đơn giản, kèm giải mã có hiệu chỉnh, đã vượt baseline tốt nhất của bài gốc khoảng 9 điểm Arg-C.

**Về HONE:** thêm một bộ thẩm định học từ ứng viên out-of-fold nâng precision rõ rệt và đưa hệ thống lên 52.36 Arg-C, hơn OneIE 10.75 điểm. Tuy vậy, với bộ đề xuất CARVE-simple, phần cộng thêm của bộ thẩm định chưa được xác lập ở mức ý nghĩa thống kê.

**Hai bài học về phương pháp luận:**
1. Suy luận kiến trúc từ thống kê dữ liệu cần được ablation kiểm chứng. Ở đây, hai quyết định như vậy của CARVE gốc đều không đứng vững.
2. Một bộ thẩm định chỉ học được điều gì đó khi dữ liệu huấn luyện của nó có lỗi thật. Cross-fitting là điều kiện cần.

**Hướng tiếp theo:**
- chạy nhiều seed cho hệ thống cuối;
- lặp lại các ablation của HONE trên CARVE-simple;
- nhắm vào khoảng trống 72 → 49 của quyết định giữ/loại.

---

## Phụ lục A. Tái lập

```bash
pip install -e .
bash scripts/setup_data.sh
python3 tests/test_contract.py
bash scripts/reproduce.sh
```

Chi tiết từng bước, tham số và các kiểm tra tương đương đã thực hiện: README và PAPER_NOTES §8.
