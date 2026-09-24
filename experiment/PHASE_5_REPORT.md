# Báo cáo Pha 5 — 2026-09-24

Chạy `bash experiment/scripts/reproduce.sh 5` từ `/home/haipd/CLAVE` với môi trường trong kế hoạch. Sáu verifier được huấn luyện đủ 5 epoch: ba run trên ứng viên in-sample (T-A1, bỏ cross-fitting), ba run OOF không dùng đặc trưng proposer (T-A2). Mỗi biến thể được chọn **luật riêng trên dev với cùng lưới** trước lần test đã đăng ký. `RUN_LOG.md` ghi đường học Arg-C của từng epoch, số ứng viên, seed và epoch tốt nhất.

## Huấn luyện và luật đóng băng

| Biến thể | Seed 42: epoch tốt nhất / dev F1 | Seed 13 | Seed 101 | Luật chung | Dev Arg-C mean ± SD |
|---|---:|---:|---:|---|---:|
| T-A1 in-sample | 5 / 45,19 | 5 / 46,15 | 2 / 46,50 | θ 0,80; min_len 3; NMS any; λ 0,75; α 0,50 | **46,80 ± 0,79** |
| T-A2 không đặc trưng | 4 / 47,48 | 2 / 48,82 | 5 / 48,35 | θ 0,30; min_len 3; NMS any; λ 0,50; α 0,50 | **49,08 ± 1,45** |
| CLAVE đầy đủ (T-main) | — | — | — | θ 0,30; min_len 3; NMS any; λ 0,50; α 0,50 | **48,43 ± 0,47** |

Luật T-A2 trùng giá trị với T-main nhưng được quét và đóng băng độc lập trên dev của ba run T-A2. T-A1 dùng các ứng viên train in-sample dương 98,11–98,64%, khác với OOF 37,02–38,54%; đây là tác động phân phối chủ ý của ablation. Các luật được đóng băng lúc **14:41:32** (T-A1) và **14:43:42** (T-A2). Dòng `TEST_LOG.md` tương ứng được ghi trước test lúc **14:41:35** và **14:43:46**. `check_test_log.py --ids T-main T-A1 T-A2` qua; mọi file dự đoán có 163 dòng và thời điểm ghi sau luật.

## Kết quả test và bootstrap

Số liệu đầy đủ ở [T5](tables/T5.md), [a1.json](results/a1.json), [a2.json](results/a2.json), [bootstrap_a1.json](results/bootstrap_a1.json), [bootstrap_a2.json](results/bootstrap_a2.json). Các hàng giải mã T-A3–A7 của T5 còn `pending` cho Pha 6.

| Hệ thống | Test Arg-I F1 | Test Arg-C P / R / F1 (mean ± SD trên 3 seed) | Δ Arg-C so với CLAVE đầy đủ, 95% CI |
|---|---:|---|---:|
| CLAVE đầy đủ | 60,49 ± 1,25 | 60,77 ± 7,27 / 46,72 ± 4,01 / **52,45 ± 0,73** | 0 |
| T-A1 bỏ cross-fitting | 58,15 ± 1,38 | 46,76 ± 0,71 / 51,41 ± 0,50 / **48,97 ± 0,61** | **−3,48 [−4,92; −2,10]** |
| T-A2 bỏ đặc trưng | 60,16 ± 2,00 | 59,18 ± 1,13 / 45,59 ± 2,45 / **51,46 ± 1,22** | **−0,99 [−2,03; +0,00]** |

CI của T-A2 chạm 0 ở độ chính xác hai chữ số, nên kết quả test này chưa cho bằng chứng rõ ràng ở mức CI 95% rằng đặc trưng cải thiện Arg-C. Bootstrap dùng 5.000 mẫu ghép cặp theo cửa sổ, lấy trung bình ba file mỗi hệ thống.

## G2 và G3 theo tiêu chí đăng ký trước

- **G2 (cross-fitting cần thiết): được ủng hộ trên dev.** CLAVE đầy đủ hơn T-A1 **+1,63** điểm dev Arg-C, vượt ngưỡng +0,5. Trên test, CLAVE hơn **+3,48**, CI 95% **[+2,10; +4,92]**.
- **G3 (đặc trưng proposer có ích): bị bác bỏ trên dev.** CLAVE đầy đủ **thấp hơn T-A2 0,66 điểm dev Arg-C**, ngược chiều giả thuyết và dưới ngưỡng +0,5. Trên test, CLAVE cao hơn **+0,99**, nhưng CI của chênh lệch là **[0,00; +2,03]** ở độ chính xác hai chữ số khi đảo dấu từ bootstrap T-A2 − CLAVE; điều này không đổi kết luận đăng ký trên dev.

Sau khi có `log.json`, `dev_probs.npy`, `test_probs.npy`, JSON kết quả và bootstrap, script đã xóa cả sáu `best.pt` ablation theo §8. Kiểm tra xác suất dev/test hữu hạn, mỗi hàng cộng xấp xỉ 1. Chỉ hai checkpoint verifier chính của Pha 3 còn trong `experiment/runs/`. Đĩa trống đo cuối pha là 136 GB; mức này tăng mạnh trong lúc chạy do thay đổi dung lượng ngoài quá trình ablation, không được quy cho việc dọn sáu checkpoint. `git status --short` chỉ hiện `?? experiment/`.

Sai sót xem nhãn test ngoài kế hoạch ở Pha 2 vẫn được giữ nguyên trong `TEST_LOG.md`; không dùng số đó để chọn luật hoặc mô hình ở Pha 5.

**STOP-5:** Dừng tại đây; chỉ bắt đầu Pha 6 sau khi người dùng nói `tiếp`.
