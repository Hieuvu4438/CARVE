# Kế hoạch thí nghiệm đầy đủ cho CLAVE

**Đối tượng đọc:** agent thực thi. Làm đúng theo thứ tự các Phase. **Không** làm gì ngoài kế hoạch này mà không hỏi người dùng.

**Mục tiêu:** tạo ra các bảng chính của phần Experiment cho paper *CLAVE: Cross-Fitted Clause-Level Argument Verification for Scientific Event Extraction*:
- kết quả chính, 3 seed, với đủ P/R/F1;
- ý nghĩa thống kê;
- ablation study (bộ thẩm định, cross-fitting, đặc trưng, các thành phần giải mã, thiết kế bộ đề xuất);
- phân tích lỗi.

Ngày lập: 2026-09-24. Kho làm việc: `/home/haipd/CLAVE` (remote `github.com/Hieuvu4438/CLAVE`).

---

## 0. Quy tắc bất biến (đọc trước khi làm bất cứ gì)

1. **Không bao giờ chỉnh tham số nào dựa trên test.** Mọi lựa chọn (checkpoint, epoch, θ, min_len, luật chồng lấn, α, λ, τ) chỉ được làm trên **dev**. Luật phải được **đóng băng thành file có timestamp** trước khi sinh bất kỳ dự đoán test nào của hệ thống đó.
2. **Chỉ các lần đánh giá test đã đăng ký trước ở §2 mới được chạy.**
   - Mỗi lần chạy test phải ghi một dòng vào `experiment/TEST_LOG.md` (mẫu ở §9).
   - Muốn thêm một biến thể test không có trong danh sách: dừng lại và hỏi người dùng. Nếu được đồng ý, phải đánh dấu dòng đó là *post-hoc*.
3. **Không sửa `third_party/SciEvent`.** Bộ chấm chính thức chỉ được import qua `clave.evaluate`, không được cài lại.
4. **Không mô hình nào được đọc `sent_id`** hay vị trí cửa sổ trong tóm tắt. Hậu tố `-k` của `sent_id` rò rỉ loại sự kiện (94–99%).
5. **Không ghi đè** các file của hệ thống đã phát hành:
   - `runs/verifier_s42/`, `runs/proposer_s{42,13,101}/`
   - `assets/decoding_rule.json`, `assets/proposer_rule.json`
   - `preds/test_verifier_s42.jsonl`, `preds/test_proposer_s*.jsonl`
6. **Không kill tiến trình của người khác trên GPU.** Nếu thiếu bộ nhớ thì chạy tuần tự.
7. **Dừng ở mỗi mốc STOP.** Ở mỗi mốc: báo cáo kết quả của phase (bảng, số, mọi sai lệch so với kỳ vọng) rồi **chờ người dùng nói "tiếp"** mới sang phase sau. Code của phase sau có thể viết trước; chỉ **huấn luyện và đánh giá** là phải chờ.
8. **Báo cáo trung thực.** Mọi con số đều kèm cấu hình và seed. Kết quả âm tính được báo cáo như kết quả. Nếu một số liệu không khớp kỳ vọng (§3), dừng lại và điều tra, không "sửa cho khớp".

### Môi trường (mọi lệnh chạy từ `/home/haipd/CLAVE`)

```bash
cd /home/haipd/CLAVE
export SCIEVENT_ROOT=/home/haipd/SciEvent/third_party/SciEvent
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Tài nguyên** (đo ngày 2026-09-24):
- GPU 48GB dùng chung. Một proposer chiếm khoảng 21GB; một verifier (có gradient checkpointing) khoảng 9GB.
- Đĩa còn khoảng **28GB trống**. Mỗi checkpoint DeBERTa-large nặng khoảng **1.7GB**; ngân sách đĩa ở §8.
- Thời gian:
  - proposer: khoảng 10 phút một lần (30 epoch × ~19 giây);
  - verifier: khoảng 10 phút;
  - suy luận sinh ứng viên: khoảng 1 phút một split.
- Lưu ý transformers ≥ 5: code đã ép trọng số fp32. Không đổi dtype.

---

## 1. Hiện trạng (những gì đã có, không chạy lại)

| thành phần | trạng thái |
|---|---|
| proposer CARVE-simple full-train, seed 42 / 13 / 101 | `runs/proposer_s{42,13,101}/best.pt` ✅ |
| luật ngưỡng của proposer (dev, trung bình 3 seed) | `assets/proposer_rule.json`: τ 0.90, min_len 3 ✅ |
| baseline CARVE-simple trên test (3 seed) | `preds/test_proposer_s*.jsonl`: Arg-C 50.73 ± 0.19 ✅ |
| ứng viên seed 42 (OOF 5 fold + dev + test) | `data/cands/{oof_k*,dev,test}_s42.jsonl` ✅ |
| verifier v42 / p42 | `runs/verifier_s42/` (best.pt, dev_probs.npy, log.json) ✅ |
| luật 1 seed đã phát hành | `assets/decoding_rule.json`: θ .20, ml 1, any, λ .5, α .25; test 52.36 ✅ |
| CARVE gốc (2 đầu + điều kiện hoá), 3 seed | dự đoán test tại `/home/haipd/SciEvent/method/SciEvent-Next/artifacts/preds/test_preds_s*.jsonl` (50.48 ± 1.15) ✅ |
| checkpoint ablation của CARVE gốc (no_type_cond, single_head, crf, span_role, llrd × 3 seed) | `/home/haipd/SciEvent/method/CARVE-full/runs/abl_*` ✅ (chỉ có số dev) |

Các lần đã chạm test trước đây được liệt kê ở `docs/PAPER_NOTES.md` §3. Kế hoạch này **bổ sung** các lần mới ở §2.

---

## 2. Đăng ký trước: hệ thống chính, ablation và các lần đánh giá test

**Thiết kế seed.** Ba cặp *(seed proposer = seed verifier)*: **(42, 42), (13, 13), (101, 101)**.
- Mỗi cặp có một pipeline riêng: 5 proposer OOF với seed S, verifier seed S, và ứng viên dev/test từ `proposer_sS`.
- Cặp (42, 42) đã có sẵn; chỉ cần tạo symlink `runs/verifier_v42_p42 → verifier_s42`, không huấn luyện lại.

**Con số chính của paper** = trung bình ± độ lệch chuẩn (n−1) trên 3 cặp, dưới **một luật chung** đóng băng trên dev với tiêu chí **trung bình dev Arg-C trên 3 run**. Luật mới ghi vào `assets/decoding_rule_3seed.json`. Con số 1 seed đã phát hành (52.36) vẫn được giữ và báo cáo riêng; không thay thế nó.

**Các lần đánh giá test mới được phép** (mỗi dòng: đóng băng trên dev trước, rồi test đúng một lần):

| ID | hệ thống | run | luật (đóng băng trên dev) |
|---|---|---|---|
| T-main | CLAVE, 3 seed | `verifier_v{42,13,101}_p*` | `decoding_rule_3seed.json` |
| T-A1 | − cross-fitting (verifier học trên ứng viên in-sample) | `verifier_insample_v*_p*` × 3 | `rule_abl_insample.json` |
| T-A2 | − đặc trưng proposer (verifier chỉ đọc văn bản) | `verifier_nofeat_v*_p*` × 3 | `rule_abl_nofeat.json` |
| T-A3..A7 | các ablation giải mã (dùng xác suất đã lưu của T-main, không huấn luyện lại): α = 1; α = 0; λ = 0; α = 1 và λ = 0; luật chồng lấn `iou`; luật chồng lấn `wis` | 3 run của T-main | mỗi biến thể một luật riêng |
| T-P1 | proposer argmax, không hiệu chỉnh | `proposer_s*` | τ = 0, min_len 1 (cố định, không dò) |
| T-P2 (tuỳ chọn) | ô còn thiếu của bảng 2×2 thiết kế proposer: `abl_no_type_cond` và `abl_single_head` | archive | trung bình dev 3 seed |

Khi người dùng duyệt T-P2, ghi hai lần nhìn test thành **T-P2a** (`abl_no_type_cond`)
và **T-P2b** (`abl_single_head`) trong `TEST_LOG.md`, mỗi lần với luật riêng đóng
băng trên dev trước test. Bản sao code archive và mọi output mới đặt trong
`experiment/archive_carve_full/` để giữ nguyên archive gốc.

**Tổng số lần nhìn test mới:** 1 (T-main) + 2 (T-A1, T-A2) + 6 (T-A3..A7) + 1 (T-P1), cộng 2 nếu chạy T-P2. Bất cứ gì ngoài danh sách này là *post-hoc* (quy tắc 0.2).

**Giả thuyết đăng ký trước.** Ngưỡng bác bỏ: Δ < +0.5 dev Arg-C, trung bình 3 seed.

| ID | giả thuyết | đối chứng |
|---|---|---|
| G1 | verifier > ngưỡng của proposer (CLAVE > CARVE-simple) | CARVE-simple, cùng 3 seed |
| G2 | cross-fitting là cần thiết | T-A1 |
| G3 | đặc trưng proposer có ích | T-A2 |
| G4 | trộn vai trò α có ích | T-A3 (α = 1) |
| G5 | loại sự kiện kết hợp λ có ích | T-A5 (λ = 0) |

Ghi mỗi giả thuyết là *được ủng hộ* hay *bị bác bỏ* trên **dev**. Test chỉ dùng để xác nhận, kèm bootstrap.

---

## 3. Số kỳ vọng để kiểm tra tỉnh táo (lệch nhiều → dừng và điều tra)

| đại lượng | kỳ vọng (từ lịch sử) |
|---|---|
| ứng viên OOF của mỗi seed | khoảng 5.0–5.1 mỗi cửa sổ, 37–39% dương (s42: 5.06, 38.0%) |
| ứng viên dev của mỗi seed | khoảng 790–800 (s42: 792) |
| dev Arg-C của verifier, luật lúc huấn luyện (chỉ verifier) | khoảng 45–48 (s42: 46.69, ở epoch 5) |
| dev Arg-C, lưới đầy đủ, 1 run | khoảng 48–50 (s42: 49.16) |
| verifier in-sample | tỉ lệ dương của ứng viên in-sample ~97%; dev thấp hơn bản OOF vài điểm (với CARVE gốc: −4.3) |
| verifier chỉ đọc văn bản | thấp hơn khoảng 0.5–1.5 (với CARVE gốc: −0.9) |
| kiểm tra hồi quy trên s42 với luật cũ | test 52.36, file dự đoán giống hệt **từng byte** `preds/test_verifier_s42.jsonl` |

---

## 4. Phase 0: kiểm tra trước khi làm

1. Chạy `python3 tests/test_contract.py`. Phải in `ALL CONTRACT TESTS PASSED`, oracle dev/test đạt 100.00.
2. Chạy hồi quy: `cp preds/test_verifier_s42.jsonl /tmp/ref_s42.jsonl`, rồi `python3 scripts/freeze_and_test.py test --runs verifier_s42`, rồi `cmp` với file vừa copy. Kết quả phải giống hệt, Arg-C 52.36.
3. Kiểm tra `nvidia-smi` và `df -h /` (cần ≥ 20GB trống, xem §8). Tạo `experiment/RUN_LOG.md` và `experiment/TEST_LOG.md`.

**STOP-0:** báo cáo kết quả 1–3.

---

## 5. Phase 1: sửa code (chưa huấn luyện gì)

Mọi thay đổi phải **giữ nguyên hành vi mặc định** của hệ thống đã phát hành. Mỗi mục có tiêu chí chấp nhận.

**C1. `scripts/freeze_and_test.py`: hỗ trợ seed proposer khác nhau giữa các run.**
- Bỏ `prop_seed_of` với assert chung. Mỗi run tự đọc seed proposer của mình từ `runs/<run>/log.json`: khoá `prop_seed`, hoặc `prop_seeds[0]` với run cũ. Mỗi run nạp ứng viên `dev_s{P}` / `test_s{P}` của chính nó.
- `freeze`: với mỗi tổ hợp của lưới, tính dev Arg-C **cho từng run trên ứng viên của run đó**, rồi lấy trung bình. Ghi `prop_seeds` theo từng run vào file luật.
- Thêm `--fix KEY=VAL` (lặp được, ví dụ `--fix role_alpha=1.0 --fix type_lambda=0`) để cố định các chiều của lưới khi làm ablation giải mã. Thêm `--nms_modes` nếu cần giới hạn luật chồng lấn.
- `test`: lưu `runs/<run>/test_probs.npy` (khi đã có file thì dùng lại, không chạy lại mô hình). Thêm `--pred_tag TAG`: file ra là `preds/test_<run>__<TAG>.jsonl`, và mặc định TAG = tên file luật không có `.json`. **Ngoại lệ:** khi dùng `decoding_rule.json` với `verifier_s42` thì giữ tên cũ `test_verifier_s42.jsonl`, để hồi quy Phase 0 vẫn giữ nguyên. In đủ P/R/F1 cho ROUGE-L, Arg-I, Arg-C, và in mean ± std khi có ≥ 2 run.
- Dựng verifier theo `args` trong log của run (`use_feats`, xem C3).
- *Chấp nhận:* hồi quy Phase 0 cho kết quả giống hệt. `freeze --runs verifier_s42` cho ra đúng luật cũ; so với `assets/decoding_rule.json`, trừ `frozen_at`.

**C2. `scripts/propose.py`: thêm chế độ `insample`.**
- `propose.py insample --seed S` dùng `runs/proposer_sS` giải mã **chính split train** và ghi `data/cands/insample_sS.jsonl`. Chỉ dùng cho ablation T-A1.
- *Chấp nhận:* 1,278 dòng; tỉ lệ dương khoảng 97% (tính bằng `candidates.label` với gold train).

**C3. `clave/verifier.py` và `scripts/train_verifier.py`: thêm `--no_feats` và `--train_cands {oof,insample}`.**
- `Verifier(..., use_feats=True)`. Khi `False` thì không tạo `self.feat`, và head nhận đầu vào 3H.
- **Khi `use_feats=True`, thứ tự khởi tạo module phải giữ nguyên** (encoder → feat → dropout → head), để luồng ngẫu nhiên của bản phát hành không đổi.
- `--train_cands insample` nạp `insample_s{prop_seed}.jsonl` thay cho 5 file OOF. Dev vẫn dùng `dev_s{prop_seed}` như bình thường.
- Ghi cả hai khoá vào `log.json` → `args`.
- *Chấp nhận:* `--smoke --epochs 1` chạy được với cả hai tuỳ chọn. Hồi quy Phase 0 vẫn giống hệt.

**C4. `scripts/proposer_baseline.py`: thêm lệnh `eval`.**
- `proposer_baseline.py eval --split {dev,test} --tau T --min_len L --tag TAG --seeds ...` ghi `preds/{split}_proposer_TAG_s*.jsonl` và in P/R/F1. Dùng cho T-P1 (argmax: `--tau 0 --min_len 1`).
- **Không** dò tham số trên test.

**C5. `experiment/scripts/`: công cụ tổng hợp.**
- `collect.py`: đọc mọi file dự đoán và luật, ghi `experiment/results/<id>.json`. Mỗi file chứa: P/R/F1 của 4 chế độ so khớp, ROUGE-L P/R/F1, mean ± std, số run, luật và timestamp.
- `tables.py`: sinh `experiment/tables/T*.md` từ các file JSON (§7).
- `analysis.py`: các phân tích ở Phase 7.
- Dùng lại `clave.evaluate.score` (trả sẵn `rolewise_iou`, `eventtype_iou`, `domain_iou`, `rougeL_domain`, `rougeL_eventtype`) và `scripts/bootstrap_ci.py`.

**STOP-1:** báo cáo diff tóm tắt, kết quả các tiêu chí chấp nhận, và kết quả hồi quy.

---

## 6. Các phase chạy thí nghiệm

### Phase 2: ứng viên cho seed 13 và 101 (huấn luyện proposer)

```bash
for S in 13 101; do python3 scripts/propose.py eval --seed $S; done        # dev/test candidates (~2 phút)
for S in 13 101; do for K in 0 1 2 3 4; do                                   # 10 proposer OOF (~100 phút tuần tự)
  python3 scripts/propose.py oof --fold $K --seed $S > logs/oof_k${K}_s${S}.log 2>&1; done; done
```

- Có thể chạy 2 fold song song nếu GPU trống (khoảng 42GB); nếu có người khác đang dùng GPU thì chạy tuần tự.
- Script tự xoá checkpoint của mỗi fold sau khi đã ghi ứng viên. Kiểm tra lại là `runs/oof_k*_s*/best.pt` không còn.
- Ghi vào `RUN_LOG.md`: số ứng viên mỗi cửa sổ và tỉ lệ dương của từng seed (so với §3).
- Chạy thêm C2 cho cả ba seed: `propose.py insample --seed {42,13,101}` (chỉ suy luận).

**STOP-2**

### Phase 3: verifier chính cho hai seed mới

```bash
ln -s verifier_s42 runs/verifier_v42_p42
for S in 13 101; do
  python3 scripts/train_verifier.py --prop_seed $S --seed $S --name verifier_v${S}_p${S} > logs/verifier_v${S}_p${S}.log 2>&1
done
```

Ghi vào `RUN_LOG.md`: đường học dev theo từng epoch và epoch tốt nhất (so với §3).

**STOP-3**

### Phase 4: kết quả chính (T-main)

```bash
python3 scripts/freeze_and_test.py freeze --runs verifier_v42_p42 verifier_v13_p13 verifier_v101_p101 --rule_name decoding_rule_3seed.json
# ghi TEST_LOG (luật, timestamp) TRƯỚC, rồi:
python3 scripts/freeze_and_test.py test   --runs verifier_v42_p42 verifier_v13_p13 verifier_v101_p101 --rule_name decoding_rule_3seed.json
python3 scripts/bootstrap_ci.py --sys 'preds/test_verifier_v*_p*__decoding_rule_3seed.jsonl' --ref 'preds/test_proposer_s*.jsonl'
python3 scripts/bootstrap_ci.py --sys 'preds/test_verifier_v*_p*__decoding_rule_3seed.jsonl' \
        --ref '../SciEvent/method/SciEvent-Next/artifacts/preds/test_preds_s*.jsonl'
```

- Bootstrap là ghép cặp theo **cửa sổ**, mỗi bên lấy trung bình 3 file. Phía proposer dùng đúng 3 seed 42 / 13 / 101, nên các cặp seed khớp nhau.
- Kiểm tra: timestamp của file luật phải **sớm hơn** timestamp của file dự đoán test.
- Sinh bảng T1–T4 (§7).

**STOP-4:** báo cáo T1–T4 và kết luận G1 (dev và test).

### Phase 5: ablation cần huấn luyện lại (T-A1, T-A2)

```bash
for S in 42 13 101; do
  python3 scripts/train_verifier.py --prop_seed $S --seed $S --train_cands insample --name verifier_insample_v${S}_p${S}
  python3 scripts/train_verifier.py --prop_seed $S --seed $S --no_feats           --name verifier_nofeat_v${S}_p${S}
done
python3 scripts/freeze_and_test.py freeze --runs verifier_insample_v42_p42 verifier_insample_v13_p13 verifier_insample_v101_p101 --rule_name rule_abl_insample.json
python3 scripts/freeze_and_test.py freeze --runs verifier_nofeat_v42_p42  verifier_nofeat_v13_p13  verifier_nofeat_v101_p101  --rule_name rule_abl_nofeat.json
# TEST_LOG, rồi test cho từng biến thể; sau khi đã có test_probs.npy thì xoá best.pt của các run ablation (§8)
```

Mỗi biến thể ablation được đóng băng luật **của riêng nó** trên dev với cùng lưới. Đó là so sánh công bằng.

**STOP-5**

### Phase 6: ablation giải mã (T-A3..A7, không huấn luyện)

Dùng `dev_probs.npy` và `test_probs.npy` của 3 run T-main. Mỗi biến thể: `freeze` với `--fix ...` → file luật riêng → `test` (đọc lại `test_probs.npy`, không chạy lại mô hình).

| biến thể | cố định |
|---|---|
| A3 chỉ vai trò của verifier | `role_alpha=1.0` |
| A4 chỉ vai trò của proposer | `role_alpha=0.0` |
| A5 không loại sự kiện kết hợp | `type_lambda=0` |
| A6 verifier thuần | `role_alpha=1.0 type_lambda=0` |
| A7a / A7b luật chồng lấn | `nms=iou` / `nms=wis` |

Các chiều còn lại được dò lại trên dev như bình thường.

**STOP-6**

### Phase 7: ablation proposer và phân tích

1. **T-P1:** chạy `proposer_baseline.py eval --split dev --tau 0 --min_len 1 --tag argmax` và `--split test`. Đây là hàng "không hiệu chỉnh".
2. **T-P2 (tuỳ chọn, hỏi người dùng trước).**

   > **Cập nhật 2026-09-24:** checkpoint ablation của CARVE gốc (`CARVE-full/runs/abl_*`) **đã bị xoá** khi dọn đĩa. Muốn làm T-P2 phải **huấn luyện lại** 6 run bằng code của archive. Cách chạy: `python3 -m carve.train -c configs/abl_no_type_cond.json --set run_name=abl_no_type_cond_s$S seed=$S` và tương tự với `configs/abl_single_head.json`, cho S = 42, 13, 101; tổng khoảng 1 giờ GPU và 10GB đĩa. Số dev sẽ lệch nhẹ so với bảng 12 (GPU không tất định). Phải hỏi người dùng trước.

   Dùng **code của archive** và checkpoint huấn luyện lại. Bản sao cách ly ở
   `/home/haipd/CLAVE/experiment/archive_carve_full` chứa `carve/`, `configs/`,
   `scripts/` và `assets/` lấy từ archive; chạy ở đó với `PYTHONPATH=.` và cùng
   `SCIEVENT_ROOT`. Mã train/model/decode và cấu hình giữ nguyên từng byte. Script
   một head trong bản sao chỉ tách `freeze` và `test` để ghi timestamp và log trước
   lần test; archive gốc vẫn chỉ đọc.

   ⚠ `freeze_rules.py` **mặc định ghi đè `assets/decoding_rules.json`**, tức luật đã đóng băng của CARVE gốc. Wrapper dưới đây truyền tên file riêng ở tham số thứ 3, lưu `frozen_at` và ghi TEST_LOG trước suy luận test.

   ```bash
   cd /home/haipd/CLAVE
   bash experiment/scripts/reproduce_p2.sh train
   bash experiment/scripts/reproduce_p2.sh test
   ```

   - Trước khi chạy, xác nhận hai file `assets/decoding_rules.json` và `assets/decoding_rules.carve_simple.json` của archive **không đổi** (lưu checksum trước và sau).
   - *Kiểm tra tỉnh táo:* dev từng seed phải khớp bảng 12 trong `docs/PAPER_NOTES.md` của archive (± sai khác do khác lưới; ghi rõ).
3. **Phân tích (T7):**
   - trên **dev**: oracle keep + role với ứng viên của từng seed (s42 = 72.20 dưới luật một seed cũ; luật ba seed mới cho 69.91), và oracle loại sự kiện;
   - trên **test**, mô tả (chỉ sau khi mọi lần test ở trên đã xong, không dùng để ra quyết định):
     - F1 theo vai trò, theo loại sự kiện, theo lĩnh vực;
     - recall theo nhóm độ dài span (1–4, 5–9, 10–19, 20–39, ≥ 40 token);
     - phân loại lỗi: sai ranh giới (IoU ≤ 0.5 nhưng có giao), sai vai trò, dự đoán thừa, bỏ sót;
     - mức dịch chuyển P/R giữa CARVE-simple và CLAVE.
4. **Thống kê ứng viên và chi phí:** số tham số, thời gian GPU mỗi giai đoạn (lấy từ log).

**STOP-7**

### Phase 8: cập nhật tài liệu

- Cập nhật `docs/PAPER_NOTES.md`: thêm các mục mới, test-look log, cả hai con số chính (1 seed đã phát hành và 3 seed).
- Cập nhật `docs/PAPER_VI.md` (bảng chính và ablation) và `README.md` (thay con số chính bằng số 3 seed, ghi rõ luật nào).
- Cập nhật `scripts/reproduce.sh` cho thiết kế 3 seed.
- **Không commit hay push khi người dùng chưa yêu cầu.**

**STOP-8**

### Phase 9 (tuỳ chọn, không làm khi chưa được đồng ý)

- **Split rời rạc theo tài liệu:** chạy lại toàn bộ pipeline trên `docsplit` (archive có `scripts/make_docsplit.py`); chi phí khoảng 4–5 giờ GPU.
- **Tổng quát hoá:** một bộ dữ liệu có luận cứ dài hoặc mức mệnh đề (ví dụ PHEE).
- **Mở rộng:** gộp ứng viên từ nhiều seed proposer (HONE-H2; với CARVE gốc đạt 53.06 / 63.05).

---

## 7. Các bảng cần tạo (`experiment/tables/`)

| bảng | nội dung | nguồn |
|---|---|---|
| **T1** | Trigger ROUGE-L P/R/F1: các baseline của bài gốc (sao từ `docs/PAPER_NOTES.md` §4.1), CARVE gốc, CARVE-simple, **CLAVE (3 seed, mean ± std)**, CLAVE 1 seed đã phát hành | T-main |
| **T2** | Arg-I và Arg-C, IoU > 0.5, P/R/F1: cùng các hàng như T1 | T-main |
| **T3** | Bốn chế độ so khớp (EM, overlap, SciREX, IoU) × P/R/F1: CLAVE so với CARVE-simple | T-main |
| **T4** | Bootstrap ghép cặp (Δ, 95% CI, P(Δ > 0)) cho Arg-C và Arg-I: CLAVE so với CARVE-simple, so với CARVE gốc; CI tuyệt đối so với OneIE | T-main |
| **T5** | **Ablation chính** (dev mean ± std; test Arg-I F1 và Arg-C P/R/F1 mean ± std; Δ và CI so với CLAVE đầy đủ). Các hàng: CLAVE đầy đủ; − verifier (= CARVE-simple); − cross-fitting (A1); − đặc trưng (A2); − trộn vai trò, α = 1 (A3); chỉ vai trò proposer, α = 0 (A4); − loại kết hợp (A5); verifier thuần (A6); luật chồng lấn iou / wis (A7) | P5, P6 |
| **T6** | **Thiết kế proposer:** argmax so với có hiệu chỉnh (T-P1); bảng 2×2 {2 đầu, 1 đầu} × {có, không điều kiện hoá} (dev 3 seed có sẵn: 47.58 / 47.34 / 48.34 / 48.50; test cho 4 ô nếu chạy T-P2); + CRF / + head mức span / + LLRD (chỉ dev, từ archive) | P7 |
| **T7** | Phân tích: theo vai trò, loại sự kiện, lĩnh vực; theo nhóm độ dài; phân loại lỗi; oracle (dev) | P7 |
| **T8** | Thống kê ứng viên của từng seed (OOF, dev, in-sample); chi phí tính toán | P2, P7 |

**Quy ước:**
- Mọi số là % với 2 chữ số thập phân; mean ± std (n−1) trên 3 seed.
- Luôn ghi tên file luật đi kèm mỗi hàng.
- Số dev và số test để ở hai cột riêng.
- Hàng nào lấy từ lịch sử hay archive thì đánh dấu *(archive)*.

---

## 8. Ngân sách đĩa và dọn dẹp

| mục | dung lượng |
|---|---|
| verifier chính mới (v13, v101) | 3.4GB, **giữ lại** |
| verifier ablation (6 run) | 6 × 1.7 = 10.2GB tạm thời |
| checkpoint OOF | tự xoá sau khi ghi ứng viên |

- Sau khi `test_probs.npy` và `dev_probs.npy` của một run ablation đã được lưu và kết quả đã ghi vào `RUN_LOG.md`, **xoá `best.pt` của run đó**. Giữ `log.json`, `dev_probs.npy`, `test_probs.npy`.
- Trước mỗi lần huấn luyện, kiểm tra `df -h /` còn ≥ 5GB. Nếu không đủ: dừng và hỏi người dùng, **không** tự xoá gì ngoài các checkpoint ablation đã được phép xoá.
- Không bao giờ xoá: `runs/proposer_s*`, `runs/verifier_s42`, `runs/verifier_v13_p13`, `runs/verifier_v101_p101`, `data/`, `assets/`, `preds/`.

---

## 9. Mẫu ghi log

`experiment/TEST_LOG.md` (một dòng cho mỗi lần đánh giá test):

```
| thời điểm | ID (§2) | runs | file luật | frozen_at của luật | file dự đoán | Arg-C (mean±std) | ghi chú (pre-registered / post-hoc) |
```

`experiment/RUN_LOG.md` (mỗi lần huấn luyện): lệnh, seed, thời gian bắt đầu và kết thúc, epoch tốt nhất, dev Arg-C, thống kê ứng viên, mọi sai lệch so với §3 cùng cách xử lý.

---

## 10. Checklist hoàn thành

- [ ] Phase 0 hồi quy giống hệt từng byte; test contract qua
- [ ] C1–C5 đạt tiêu chí chấp nhận; hồi quy vẫn giống hệt
- [ ] ứng viên cho 3 seed; OOF khớp §3
- [ ] 3 verifier chính; luật `decoding_rule_3seed.json` đóng băng trước khi test
- [ ] T-main, T-A1..A7, T-P1 (và T-P2 nếu được duyệt) đều có dòng trong TEST_LOG, không có lần test nào ngoài danh sách
- [ ] bảng T1–T8 được sinh bằng script từ `experiment/results/*.json`, không gõ tay
- [ ] kết luận G1–G5, mỗi cái ghi rõ dev và test cùng CI
- [ ] tài liệu được cập nhật; chưa commit khi người dùng chưa yêu cầu
