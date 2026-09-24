# Báo cáo Pha 4 — 2026-09-24

Đã chạy `bash experiment/scripts/reproduce.sh 4` từ `/home/haipd/CLAVE` với môi trường trong kế hoạch. Lệnh đóng băng luật chung trên dev của ba run `(42,42)`, `(13,13)`, `(101,101)`, sau đó chạy đúng lần test T-main đã đăng ký, tổng hợp các baseline lịch sử, bootstrap ghép cặp 5.000 mẫu và sinh T1–T4. Không điều chỉnh tham số theo test.

## Luật và kiểm toán test

`assets/decoding_rule_3seed.json` trong `experiment/` được đóng băng lúc **13:45:39** ngày 2026-09-24: `theta=0.30`, `min_len=3`, `nms=any`, `type_lambda=0.50`, `role_alpha=0.50`. Luật chọn bằng mean dev Arg-C **48,43 ± 0,47**, với từng run lần lượt **48,97 / 48,17 / 48,14**. Luật ghi đúng `prop_seeds=[42,13,101]`.

Dòng T-main trong `TEST_LOG.md` được ghi trước khi đọc/chấm test, ghim SHA-256 `aa1901e502af88a23a6e585bcbae3e32687220b161e5d2cae650e5a96a30014b`. Cả ba file dự đoán test có đúng 163 dòng và thời điểm ghi sau file luật; `check_test_log.py --ids T-main` qua. Seed 42 dùng lại xác suất test đã phát hành; seed 13/101 lưu `test_probs.npy` trong `experiment/runs/`. Không có biến thể test khác trong Pha 4. Luật của baseline archive không có `frozen_at`, nên thời điểm đóng băng của chúng không thể kiểm chứng từ file archive hiện có.

## T1–T4

Các bảng đầy đủ được sinh bằng script: [T1](tables/T1.md) (Trigger ROUGE-L), [T2](tables/T2.md) (Arg-I/Arg-C IoU), [T3](tables/T3.md) (bốn chế độ so khớp), [T4](tables/T4.md) (bootstrap). Không có ô `pending` trong T1–T4.

| Chỉ số test, mean ± SD qua 3 seed | CLAVE | CARVE-simple | CARVE gốc |
|---|---:|---:|---:|
| Trigger ROUGE-L F1 | 77,22 ± 0,81 | 77,22 ± 0,81 | 76,93 ± 0,80 |
| Arg-I IoU F1 | **60,49 ± 1,25** | 58,58 ± 0,57 | 57,95 ± 2,46 |
| Arg-C IoU F1 | **52,45 ± 0,73** | 50,73 ± 0,19 | 50,48 ± 1,15 |

Arg-C của ba run CLAVE là **53,25 / 51,80 / 52,31** cho seed 42 / 13 / 101. Theo T3, Arg-C F1 của CLAVE so với CARVE-simple lần lượt là exact **36,74 vs 34,23**, overlap **59,96 vs 59,85**, SciREX **56,21 vs 56,26**, IoU **52,45 vs 50,73**. Vì vậy cải thiện không đồng đều ở mọi chế độ so khớp.

Bootstrap ghép cặp theo 163 cửa sổ, mỗi bên lấy trung bình ba file:

| So sánh | Chỉ số | Δ F1 | 95% CI | P(Δ > 0) từ bootstrap |
|---|---|---:|---:|---:|
| CLAVE − CARVE-simple | Arg-C IoU | **+1,72** | **[+0,24; +3,30]** | 0,989 |
| CLAVE − CARVE-simple | Arg-I IoU | +1,90 | [+0,21; +3,72] | 0,984 |
| CLAVE − CARVE gốc | Arg-C IoU | +1,97 | [−0,20; +4,07] | 0,961 |
| CLAVE − CARVE gốc | Arg-I IoU | +2,54 | [+0,45; +4,65] | 0,991 |

CI Arg-C so với CARVE gốc cắt 0; kết quả đó chưa xác lập chênh lệch dương ở mức CI 95%.

## G1 theo tiêu chí đăng ký trước

Baseline CARVE-simple có **dev Arg-C 48,10 ± 0,96** trong luật đã phát hành. CLAVE đạt **48,43 ± 0,47**, chênh **+0,33 điểm**, nhỏ hơn ngưỡng **+0,5** đã đăng ký ở §2. **G1 bị bác bỏ trên dev theo tiêu chí đăng ký trước.** Test cho thấy chênh lệch dương **+1,72**, CI bootstrap **[+0,24; +3,30]**. Kết quả test được báo cáo như xác nhận mô tả, không dùng để đảo quyết định G1 trên dev.

`git status --short` chỉ hiện `?? experiment/`; đĩa còn 24 GB trống. Sai sót xem nhãn test ngoài kế hoạch ở Pha 2 vẫn được giữ nguyên trong `TEST_LOG.md` và không được dùng trong Pha 4.

**STOP-4:** Dừng tại đây; chỉ bắt đầu Pha 5 sau khi người dùng nói `tiếp`.
