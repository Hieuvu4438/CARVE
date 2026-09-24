# Báo cáo Pha 6 — 2026-09-24

Chạy `bash experiment/scripts/reproduce.sh 6` từ `/home/haipd/CLAVE` với môi trường trong kế hoạch. Không huấn luyện mô hình: cả sáu ablation T-A3, T-A4, T-A5, T-A6, T-A7a, T-A7b dùng lại `dev_probs.npy` và `test_probs.npy` của ba run T-main. Mỗi biến thể cố định đúng chiều đăng ký ở §6, dò lại các chiều còn lại trên dev bằng cùng lưới, lưu luật riêng có timestamp, rồi mới chạy lần test đã đăng ký.

## Kết quả

| Biến thể | Ràng buộc cố định | Dev Arg-C F1 mean ± SD | Test Arg-C F1 mean ± SD | Δ test so với CLAVE đầy đủ, 95% CI |
|---|---|---:|---:|---:|
| CLAVE đầy đủ | — | 48,43 ± 0,47 | **52,45 ± 0,73** | 0 |
| T-A3, chỉ vai trò verifier | α=1 | 47,36 ± 1,08 | **50,00 ± 1,58** | **−2,46 [−4,35; −0,56]** |
| T-A4, chỉ vai trò proposer | α=0 | 48,06 ± 0,59 | **52,16 ± 0,31** | −0,29 [−0,73; +0,08] |
| T-A5, bỏ kết hợp loại sự kiện | λ=0 | 48,35 ± 0,34 | **52,30 ± 0,82** | −0,15 [−0,48; +0,00] |
| T-A6, verifier thuần | α=1; λ=0 | 47,14 ± 1,07 | **49,62 ± 1,23** | **−2,83 [−4,64; −1,00]** |
| T-A7a, NMS iou | nms=iou | 48,43 ± 0,47 | **52,45 ± 0,73** | 0,00 [0,00; 0,00] |
| T-A7b, NMS wis | nms=wis | 48,43 ± 0,47 | **52,45 ± 0,73** | 0,00 [0,00; 0,00] |

P/R/F1 đầy đủ cho Arg-I và Arg-C, tên luật và các hàng T-A1/T-A2 nằm trong [T5](tables/T5.md). Các khoảng tin cậy dùng 5.000 mẫu bootstrap ghép cặp theo 163 cửa sổ, so sánh trung bình ba seed. Kết quả máy đọc được lưu ở `results/a3.json` đến `results/a7_wis.json` và `results/bootstrap_a*.json` tương ứng.

## G4 và G5 theo tiêu chí đăng ký trước

- **G4 (trộn vai trò α có ích): được ủng hộ trên dev.** CLAVE đầy đủ hơn T-A3 **+1,07 điểm dev Arg-C**, vượt ngưỡng +0,5. Trên test, mức hơn là **+2,46 điểm**, CI 95% **[+0,56; +4,35]**.
- **G5 (kết hợp loại sự kiện λ có ích): bị bác bỏ trên dev.** CLAVE đầy đủ hơn T-A5 chỉ **+0,08 điểm dev Arg-C**, dưới ngưỡng +0,5. Trên test, mức hơn là **+0,15 điểm**; CI của T-A5 − CLAVE là **[−0,48; +0,00]** ở độ chính xác báo cáo, nên không có bằng chứng rõ ràng về chênh lệch test. Kết quả test không đổi kết luận đăng ký trên dev.

T-A4 chỉ thấp hơn CLAVE đầy đủ 0,36 điểm dev và 0,29 điểm test; CI test cắt 0. T-A6 giảm rõ hơn, 1,29 điểm dev và 2,83 điểm test. Hai luật NMS cho đúng cùng điểm dev/test với luật `any`. T-A7a cho ba file dự đoán giống T-main **từng byte**. T-A7b thay thứ tự/ID thực thể ở một số cửa sổ nhưng tập *(loại sự kiện, vai trò, span)* của từng cửa sổ giống T-main ở cả ba seed; đó là lý do mọi chỉ số và bootstrap delta bằng 0.

## Kiểm toán

`check_test_log.py` xác nhận T-main, T-A1, T-A2 và cả sáu ID Pha 6 đều có một dòng hoàn tất. Mỗi luật T-A3–A7 ghi đúng ràng buộc cố định và `frozen_at`; cả 18 file dự đoán test có 163 dòng, được ghi sau luật tương ứng. Log của cả sáu lần test xác nhận **dùng lại xác suất test đã lưu**, không suy luận mô hình mới. `git status --short` chỉ hiện `?? experiment/`; đĩa còn 135 GB. Sai sót xem nhãn test ngoài kế hoạch ở Pha 2 vẫn được lưu trong `TEST_LOG.md` và không dùng để chọn các luật Pha 6.

**STOP-6:** Dừng tại đây; chỉ bắt đầu Pha 7 sau khi người dùng nói `tiếp`.
