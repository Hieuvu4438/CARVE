# Báo cáo Pha 3 — 2026-09-24

Chạy `bash experiment/scripts/reproduce.sh 3` từ `/home/haipd/CLAVE` với môi trường trong kế hoạch. Cả hai verifier chính được huấn luyện đủ 5 epoch trên ứng viên OOF của proposer cùng seed. Checkpoint được chọn theo dev Arg-C IoU F1; chưa đóng băng luật chung hoặc đánh giá test.

| Run (seed verifier/proposer) | Ứng viên train (dương) | Ứng viên dev | Dev Arg-C F1 theo epoch 1→5 | Epoch tốt nhất | Dev Arg-C F1 tốt nhất |
|---|---:|---:|---|---:|---:|
| `verifier_v13_p13` (13/13) | 6.402 (38,54%) | 815 | 48,02 → 45,61 → 46,83 → 46,74 → 46,27 | 1 | **48,02** |
| `verifier_v101_p101` (101/101) | 6.439 (37,02%) | 801 | 39,56 → 48,37 → 46,33 → 46,35 → 45,02 | 2 | **48,37** |

Mức tốt nhất hơi cao hơn khoảng kiểm tra 45–48 ở §3: +0,02 và +0,37 điểm, không phải sai lệch lớn. Epoch đầu của seed 101 thấp (39,56) nhưng hồi phục ở epoch 2; không thay đổi tham số hay kéo dài huấn luyện theo kết quả này. Các luật `theta/min_len/nms` trong `log.json` chỉ là luật chọn checkpoint trên dev, không phải luật chung đã đóng băng cho kết quả test.

Hai run dùng cùng công thức huấn luyện mặc định của verifier đã phát hành: DeBERTa-v3-large, 5 epoch, batch 8, tích lũy 2, encoder LR 1e-5, head LR 1e-4, dropout 0,1, gradient checkpointing và đặc trưng proposer. So với log seed 42 đã phát hành, các giá trị chung không đổi; khác tên run, seed và các khóa log mới `prop_seed`, `train_cands=oof`, `use_feats=true`.

Đã kiểm tra cả hai `best.pt` (mỗi file 1.748.686.321 byte), `log.json` (5 epoch) và `dev_probs.npy` (815×10 cho seed 13, 801×10 cho seed 101). Xác suất đều hữu hạn, mỗi hàng cộng xấp xỉ 1. Đường học, thời gian, epoch tốt nhất và thống kê ứng viên đã ghi trong `RUN_LOG.md`. Đĩa còn 24 GB, GPU còn khoảng 45 GB sau khi chạy; `git status --short` chỉ có `?? experiment/`.

Kế hoạch đề xuất symlink `runs/verifier_v42_p42 → verifier_s42`. Do quy tắc chỉ thêm file trong `experiment/`, mã `experiment/scripts/common.py` ánh xạ tên `verifier_v42_p42` tới run seed 42 đã phát hành dưới dạng chỉ đọc; không tạo symlink trong kho gốc. Pha 3 không có lần xem test mới; sai sót chẩn đoán test của Pha 2 vẫn được giữ nguyên trong `TEST_LOG.md`.

**STOP-3:** Dừng tại đây; chỉ bắt đầu đóng băng luật chung và đánh giá Pha 4 sau khi người dùng nói `tiếp`.
