# Báo cáo Pha 7 — 2026-09-24

Pha 7 gồm ba phần:
1. **T-P1**: proposer argmax, không hiệu chỉnh.
2. **T-P2a / T-P2b**: hai ô còn thiếu của bảng 2×2 thiết kế proposer, huấn luyện lại bằng code archive trong bản sao cô lập `experiment/archive_carve_full/`.
3. Các phân tích mô tả T7 và T8.

Mỗi lần nhìn test đều có luật đóng băng trên dev trước và một dòng trong `TEST_LOG.md`. `check_test_log.py` xác nhận đủ 12 ID đã đăng ký (T-main, T-A1…T-A7b, T-P1, T-P2a, T-P2b); không có lần test nào ngoài danh sách §2. Ngoài ra TEST_LOG còn hai dòng hồi quy (P0, P1) và một sai sót đã công bố ở Pha 2: xem nhãn test của ứng viên, không dùng cho bất kỳ lựa chọn nào.

## 1. T-P1: hiệu chỉnh giải mã so với argmax

| giải mã CARVE-simple (3 seed) | dev Arg-C | test Arg-I F1 | test Arg-C F1 |
|---|---:|---:|---:|
| argmax (τ = 0, min_len 1, cố định) | 38.32 ± 0.25 | 47.87 ± 1.31 | 39.67 ± 0.76 |
| ngưỡng hiệu chỉnh (τ 0.90, min_len 3) | 48.10 ± 0.96 | 58.58 ± 0.57 | 50.73 ± 0.19 |

- Bootstrap ghép cặp, argmax so với hiệu chỉnh: Arg-C **−11.07 [−13.36, −8.85]**, Arg-I −10.72 [−13.35, −8.08].
- Hiệu chỉnh là thành phần lớn nhất của proposer (+9.8 dev, +11.1 test), khớp với kết quả của CARVE gốc (+9.12 dev).

## 2. T-P2: bảng 2×2 thiết kế proposer

| thiết kế (3 seed, mỗi hàng một luật chung đóng băng trên dev) | dev Arg-C | test Arg-I F1 | test Arg-C F1 | Δ Arg-C so với CARVE-simple [95% CI] |
|---|---:|---:|---:|---:|
| 2 đầu + điều kiện hoá (CARVE gốc, archive) | 47.14 ± 0.26 | 57.95 ± 2.46 | 50.48 ± 1.15 | −0.25 [−2.12, +1.66] |
| 2 đầu, không điều kiện hoá (**T-P2a**) | 47.13 ± 1.32 | 57.28 ± 0.73 | 49.82 ± 0.62 | −0.91 [−2.73, +0.97] |
| 1 đầu + điều kiện hoá (**T-P2b**) | 48.59 ± 3.13 | 58.48 ± 0.56 | 50.76 ± 0.39 | +0.03 [−1.41, +1.51] |
| 1 đầu, không điều kiện hoá (**CARVE-simple**) | 48.10 ± 0.96 | 58.58 ± 0.57 | 50.73 ± 0.19 | — |

**Kết luận:** trên test không ô nào khác CARVE-simple; mọi khoảng tin cậy đều phủ 0. Kết quả này xác nhận trên test điều mà các ablation dev đã cho thấy: cả tách hai đầu lẫn điều kiện hoá đều trơ. CARVE-simple là thiết kế đơn giản nhất, không kém ô nào, và có độ lệch chuẩn test thấp nhất.

**Luật đóng băng:**
- T-P2a: τ 0.90, min_len 3.
- T-P2b: τ 0.88, min_len 3.

**Ghi chú:**
- Dòng CARVE gốc dùng checkpoint của archive; T-P2a/b huấn luyện lại. GPU không tất định, nên số dev khác nhẹ so với bảng 12 của archive. Ví dụ T-P2b seed 42 được 51.57 lúc chọn epoch, so với 50.10 trong archive.
- Mã dùng để huấn luyện giống hệt archive (`results/p2_code_provenance.json`). Riêng `freeze_and_test_simple.py` chỉ được tách thành hai bước `freeze` / `test`; logic chọn luật giữ nguyên (đã diff).
- Checksum các luật đóng băng gốc của archive vẫn OK trước và sau.

**Sự cố khi chạy (đã ghi trong TEST_LOG):**
- Lượt test đầu của T-P2a bị crash sau khi chấm seed 42, vì bộ decode của archive ghi `metrics.test.json` vào thư mục `preds/` chưa tồn tại. Chưa có file dự đoán nào được ghi.
- Lượt đó được chạy lại đến hết với **cùng luật và cùng checkpoint**, nên vẫn là cùng lần nhìn đã đăng ký.
- Máy không có `rg`, nên phép kiểm tra trùng đăng ký trong `reproduce_p2.sh` bị bỏ qua âm thầm. Đã thay bằng `grep`.

**Dọn đĩa:** sau khi ghi kết quả, đã xoá 6 `best.pt` của T-P2 (9.8GB), giữ `log.json` và `dev_preds.jsonl`. Dự đoán test nằm ở `archive_carve_full/preds/`.

## 3. Phân tích mô tả (T7, test; không dùng để ra quyết định)

**Dịch chuyển P/R** từ CARVE-simple sang CLAVE (trung bình 3 seed):

| chỉ số | ΔP | ΔR | ΔF1 |
|---|---:|---:|---:|
| Arg-C | +6.52 | −0.94 | +1.72 |
| Arg-I | +7.30 | −1.06 | +1.90 |

CLAVE thắng nhờ precision: verifier là một bộ lọc span tốt hơn ngưỡng.

**Điểm vận hành khác nhau giữa các seed.** Dưới cùng luật (θ 0.3), Arg-C của các seed là:

| seed | P / R / F1 |
|---|---|
| 42 | 64.36 / 45.40 / 53.25 |
| 13 | 52.40 / 51.22 / 51.80 |
| 101 | 65.54 / 43.53 / 52.31 |

- Verifier seed 13 được chọn ở **epoch 1** (dev 48.02), khi chưa hiệu chỉnh xong, nên nó giữ nhiều span hơn: 890 luận cứ dự đoán, so với 751 và 733.
- F1 vẫn ổn định (±0.73), nhưng P/R có độ lệch chuẩn lớn (±7.27 / ±4.01). Điều này phải ghi chú dưới bảng T2. Đây là đặc tính thật của hệ thống: epoch chọn trên dev ảnh hưởng tới hiệu chỉnh của điểm giữ.

**Điểm yếu** (từ T7):
- **Span ngắn:** recall chỉ 15.2% với span vàng dài 1–4 token và 37.8% với 5–9 token, so với khoảng 58% với 10–39 token.
- **Lĩnh vực:** Digital Humanities yếu nhất (Arg-C 33.77), bioinfo mạnh nhất (78.77).
- **Loại sự kiện:** Conclusions/Implications yếu nhất (Arg-C 37.88).
- **Vai trò:** Context có recall thấp (28.9%). Analysis, Contradictions và Ethical đạt 0 vì quá hiếm (lần lượt 10, 1, 1 gold span trong test).
- **Phân loại lỗi** (trung bình mỗi run, trên 533 gold):

  | loại | số lượng |
  |---|---:|
  | đúng | 246.7 |
  | bỏ sót | 198.3 |
  | thừa | 82.3 |
  | sai ranh giới | 50.7 |
  | sai vai trò | 37.3 |

  **Bỏ sót là lỗi chiếm ưu thế.** Phân loại này dùng phép gán một-đối-một cục bộ, còn P/R/F1 dùng bộ chấm chính thức.

**Oracle trên dev** (ứng viên của từng seed, cùng bộ giải mã):

| seed | luật thật | oracle keep + role | oracle loại sự kiện | oracle cả ba |
|---|---:|---:|---:|---:|
| 42 | 48.97 | 69.91 | 52.86 | 78.93 |
| 13 | 48.17 | 73.47 | 52.12 | 80.67 |
| 101 | 48.14 | 72.44 | 51.86 | 80.39 |

- Khoảng trống lớn nhất nằm ở quyết định **giữ/loại và gán vai trò** (khoảng +22 đến +25).
- Oracle loại sự kiện chỉ thêm khoảng +4.

## 4. Thống kê ứng viên và chi phí (T8)

**Ứng viên** (mọi số khớp kỳ vọng §3):

| tập | mỗi cửa sổ | tỉ lệ dương |
|---|---:|---:|
| OOF | 5.01–5.06 | 37.0–38.5% |
| dev | 5.01–5.16 | 42.2–42.8% |
| in-sample | 3.05–3.07 | **98.1–98.6%** |

Tỉ lệ dương gần như tuyệt đối của ứng viên in-sample là lý do định lượng cho cross-fitting.

**Chi phí, cho mỗi seed:**

| giai đoạn | thời gian |
|---|---|
| proposer full-train | khoảng 9 phút |
| 5 proposer OOF | khoảng 40–58 phút |
| verifier | khoảng 10 phút |

**Tham số:** proposer 435.1M, verifier 437.1M, tổng 872.2M.

## 5. Tổng kết các giả thuyết đăng ký trước

Ngưỡng bác bỏ: Δ dev < +0.5, trung bình 3 seed. Test chỉ dùng để xác nhận.

| ID | giả thuyết | Δ dev | kết luận (dev) | Δ test [95% CI] |
|---|---|---:|---|---:|
| G1 | verifier > ngưỡng proposer | +0.33 | **bị bác bỏ** (dưới ngưỡng) | **+1.72 [+0.24, +3.30]** |
| G2 | cross-fitting là cần thiết | +1.63 | **được ủng hộ** | +3.48 [+2.10, +4.92] |
| G3 | đặc trưng proposer có ích | −0.66 | **bị bác bỏ** (ngược chiều) | +0.99 [−0.00, +2.03] |
| G4 | trộn vai trò α có ích | +1.07 | **được ủng hộ** | +2.46 [+0.56, +4.35] |
| G5 | loại sự kiện kết hợp λ có ích | +0.08 | **bị bác bỏ** | +0.15 [−0.00, +0.48] |

**Cách đọc trung thực:**
- **G1:** với 3 seed, lợi ích của verifier so với proposer **có ý nghĩa trên test**. Nhưng mức tăng trên dev (+0.33) nằm dưới ngưỡng đã đăng ký, nên theo đúng tiêu chí đã cam kết, G1 được ghi là *bị bác bỏ trên dev, được xác nhận trên test*. Phải báo cáo cả hai.
- **G2 và G4:** vững trên cả dev lẫn test.
- **G3 và G5:** không được ủng hộ. Hai thành phần tương ứng (đặc trưng proposer và λ) giữ lại trong hệ thống không làm hại, nhưng không được coi là đóng góp.

**STOP-7.** Pha 8 (cập nhật tài liệu) được làm tiếp theo yêu cầu "hoàn thiện nốt" của người dùng. Không commit.
