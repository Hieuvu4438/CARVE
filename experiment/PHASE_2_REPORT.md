# Báo cáo Pha 2 — 2026-09-24

Chạy `bash experiment/scripts/reproduce.sh 2` từ `/home/haipd/CLAVE` với môi trường trong kế hoạch. Lệnh kết thúc thành công. Đã tạo ứng viên dev/test từ proposer full-train của seed 13 và 101, huấn luyện 10 proposer OOF (5 fold × 2 seed), sinh ứng viên in-sample cho seed 13/101, và tính thống kê tại `results/candidate_stats_phase2.json`. In-sample seed 42 từ Pha 1 được dùng lại.

## Thống kê ứng viên

| Seed | OOF train: ứng viên / cửa sổ / dương | Dev: ứng viên | In-sample train: ứng viên / dương |
|---|---:|---:|---:|
| 42 (đã có) | 6.468 / 5,061 / 38,03% | 792 | 3.897 / 98,46% |
| 13 | 6.402 / 5,009 / 38,54% | 815 | 3.917 / 98,11% |
| 101 | 6.439 / 5,038 / 37,02% | 801 | 3.901 / 98,64% |

Mỗi tập OOF và in-sample có đúng 1.278 cửa sổ train; mỗi tập dev có 158 cửa sổ. Ứng viên test của seed 13 và 101 có 163 dòng, 895 ứng viên mỗi seed; đây là số lượng file, không phải điểm đánh giá. Các bộ OOF của từng seed khớp chính xác năm tập held-out, không trùng `sent_id`, và mỗi ứng viên ghi đúng seed proposer. Cả 10 `runs/oof_k*_s*/best.pt` tạm đã được xóa sau khi ghi ứng viên; `log.json` và ứng viên được giữ.

Số OOF gộp đều đạt khoảng 5,0–5,1 ứng viên/cửa sổ và 37–39% dương ở §3. Seed 101 nằm sát mép dưới về tỷ lệ dương (37,02%); ba fold đầu chỉ 35,60%, nhưng hai fold cuối kéo mức gộp vào khoảng kỳ vọng. Dev seed 13 có 815 ứng viên, cao hơn khoảng 790–800 dự kiến 15 ứng viên; seed 101 có 801, cao hơn 1. Tất cả 158 `sent_id` dev đều đủ, không trùng; cùng mã trích xuất và proposer seed tương ứng được dùng, nên hiện không có bằng chứng file hỏng hoặc lệch split. Tỷ lệ dương in-sample 98,11–98,64% cao hơn mức xấp xỉ 97% trong kế hoạch khoảng 1–1,6 điểm phần trăm; điều này được ghi nhận, không hiệu chỉnh dữ liệu để khớp kỳ vọng.

## Huấn luyện và tài nguyên

Mỗi proposer OOF chạy đủ 30 epoch, chọn checkpoint theo dev. Epoch tốt nhất và dev Arg-C F1 của 10 run nằm trong `RUN_LOG.md`; seed 13 đạt 45,03–49,41, seed 101 đạt 44,49–49,09. Mỗi dòng run ghi seed, thời gian, epoch tốt nhất, dev F1, số ứng viên held-out và tỷ lệ dương. Huấn luyện chạy tuần tự trên GPU dùng chung. Sau khi hoàn tất, đĩa còn 28 GB trống và GPU còn khoảng 45 GB; `git status --short` chỉ hiện `?? experiment/`.

## Sai sót về nhãn test

Trong lúc kiểm tra độ phủ seed 13, một lệnh chẩn đoán đã **vô ý gán nhãn 895 ứng viên test của 163 cửa sổ và tính tỷ lệ dương 40,22%**. Đây là lần xem nhãn test ngoài danh sách đăng ký, đã ghi rõ là *post-hoc mistake* trong `TEST_LOG.md`. Không có dự đoán nào được chấm trong lệnh đó và số này không được dùng để chọn checkpoint, tham số hoặc quyết định thí nghiệm. Các kiểm tra ứng viên sau đó chỉ dùng nhãn train/dev. Đây là sai lệch quy trình cần giữ trong hồ sơ nghiên cứu, không được coi là một kết quả chính.

**STOP-2:** Dừng tại đây; chỉ bắt đầu Pha 3 sau khi người dùng nói `tiếp`.
