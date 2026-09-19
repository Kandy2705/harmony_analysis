# HARMONY Handover Analysis

Công cụ hậu xử lý dữ liệu thí nghiệm **indoor–outdoor localization handover**
(GPS ↔ VPS, qua PDR) cho hệ thống HARMONY. Recursive-scan một thư mục gốc chứa
hàng trăm trial, ghép `events_*.csv` + `samples_*.csv` + `summary_*.csv` theo
`session_id` (đọc **bên trong** file CSV, không phụ thuộc tên thư mục), **tái
tạo** các KPI quan trọng từ dữ liệu thô thay vì tin mù quáng `summary.csv`, và
xuất CSV/Excel/biểu đồ sẵn sàng dùng cho paper.

---

## 1. Vì sao không tin `summary.csv`?

Với file mẫu bạn cung cấp (`20260906_073558_794_V1_GPS_TO_VPS_NORMAL_DEV_7E24`):

- `final_state = IndoorVps`
- nhưng event log ghi rõ:
  ```
  event=handover_completed_approximate, to_state=IndoorVps
  note="30s VPS timeout; PDR pose projected to nearest B9 NavMesh point (0,87m);
        Indoor navigation continued from PDR approximate pose; VPS was not accepted"
  ```

Nếu chỉ nhìn `final_state` hay tin `summary.handover_success`, ta có thể kết
luận sai. Trial mẫu này **may mắn** đã có `summary.handover_success=0` đúng,
nhưng tool **không** dựa vào sự may mắn đó — nó luôn tái tạo độc lập từ
events + samples, rồi mới đối chiếu (cross-check) với summary. Nếu hai giá trị
khác nhau, tool **giữ giá trị tái tạo làm kết quả chính**, ghi cảnh bá
`[METRIC_MISMATCH]`, và lưu cả hai để bạn kiểm tra (`*_reconstructed` +
`*_summary` trong `per_trial_metrics.csv`).

## 2. Quy tắc phát hiện handover (xem thêm `config.yaml`)

Cột `from_state` / `to_state` / `source` trong `events_*.csv` **đổi ý nghĩa
tùy theo `event`** — đây là điểm dễ hiểu sai nhất của logger này:

| event | ý nghĩa of from_state/to_state |
|---|---|
| `outdoor_state`, `indoor_state` | trạng thái FSM ngoài/trong nhà |
| `source_switched` | nguồn cũ → nguồn mới (`Gps→Pdr`...) |
| `reliability_state` | vùng cũ → vùng mới (`OutdoorGps→EnteringWithPdr`...) |
| `vps_state` | sub-state VPS (`StartingVps→Scanning→IndoorLocalized`) |

**GPS → VPS:**
- Bắt đầu đếm giờ: `reliability_state` → `VpsScanning` (vào vùng handover thật).
- **Thành công thật** chỉ khi đạt `vps_state → IndoorLocalized` **và không**
  đi kèm bất kỳ event nào trong `events.vps_fallback_events`
  (`pdr_approximate_localization`, `handover_completed_approximate`,
  `pdr_approximate_handover_completed`) hoặc note chứa các pattern trong
  `fallback_note_patterns` (`"vps timeout"`, `"not accepted"`,...).
- Nếu timeout → fallback PDR-approximate: **luôn tính là FAILURE**, dù
  `to_state`/`final_state` là `IndoorVps`.

**VPS → GPS:**
- Bắt đầu: `reliability_state → OutdoorGps`.
- Kết thúc/thành công: `source_switched` với `to_state == Gps` sau mốc bắt đầu.

**False/Premature handover:** một `source_switched` A→B bị coi là false nếu
bị revert B→A trong vòng `oscillation.revert_window_s` giây (mặc định 8s) —
rule tường minh, cấu hình được, không suy diễn mơ hồ.

**Oscillation:** ≥ `oscillation.min_switches` lần source switch trong cửa sổ
trượt `oscillation.window_s` giây → 1 episode (đếm rời rạc, không chồng lấp).

**Position/heading jump:** cột `position_jump_m`/`heading_jump_deg` sẵn có
trong `samples_*.csv` **luôn = 0** trong dữ liệu mẫu (stub logger) → tool
**không dùng** cột này làm kết quả chính. Thay vào đó tool thử tái tạo từ tọa
độ (`campus_x/y`, `map_x/y`, `heading_deg`) trong cửa sổ ±2s quanh thời điểm
acceptance; nếu tọa độ cũng chỉ toàn 0 hoặc thiếu dữ liệu →
`NaN` + ghi rõ `NOT_EVALUABLE_FROM_CURRENT_LOG`.

## 3. Những gì tính được / chưa tính được từ logger hiện tại

**Tính chắc chắn được:** HSR, FHR (theo rule cấu hình), handover latency (khi
evaluable), VPS localization time & attempts, source switching/oscillation,
PDR transition duration, GPS/PDR statistics cơ bản.

**Chưa đủ dữ liệu (từ mẫu hiện có):** position/heading jump thực (cột stub =
0; chỉ có 1 trial mẫu, không có case VPS chấp nhận thật để kiểm chứng công
thức); VPS reliability/confidence "thật" (đều = 0 trong mẫu vì VPS chưa từng
được accept ở trial này); mọi thứ liên quan hướng VPS→GPS (không có trial mẫu
hướng này). Những trường này sẽ tự động là `NaN` kèm ghi chú khi log không đủ
— **không bịa số**.

## 4. Cài đặt

```bash
cd harmony_analysis
python3 -m venv .venv && source .venv/bin/activate   # tuỳ chọn
pip install -r requirements.txt
```

## 5. Chạy CLI

```bash
python analyze_experiments.py /path/to/HARMONY_Experiments
python analyze_experiments.py /path/to/HARMONY_Experiments --output ./analysis_output
python analyze_experiments.py /path/to/HARMONY_Experiments --exclude-invalid
python analyze_experiments.py /path/to/HARMONY_Experiments --group-by harmony_version direction scenario
python analyze_experiments.py /path/to/HARMONY_Experiments --config my_config.yaml
python analyze_experiments.py /path/to/HARMONY_Experiments --no-recursive
```

## 6. Chạy GUI

```bash
python gui.py
```

`[Select Experiment Folder]` → `[Select Output Folder]` → `[Analyze]` →
progress bar + số trial VALID/WARNING/INVALID → `[Open Output Folder]`.
GUI chỉ gọi `harmony_analysis.pipeline.run_analysis()` — toàn bộ logic phân
tích nằm trong package `harmony_analysis/`, độc lập và test được riêng.

## 7. Output (`analysis_output/`)

```
analysis_output/
├── per_trial_metrics.csv     # 1 trial = 1 row, đầy đủ metadata + KPI reconstructed
├── aggregate_results.csv     # group theo harmony_version × direction [× scenario]
├── paper_metrics.csv         # bảng sạch cho paper: HSR, FHR, latency, switching...
├── validation_report.csv     # validation_status + validation_notes mọi trial
├── event_diagnostics.csv     # inventory event/state/source names mỗi trial (audit taxonomy)
├── report_summary.xlsx       # 9 sheet: README/Per Trial/Aggregate/Paper Metrics/
│                              #   Validation/Source Transitions/GPS/PDR/VPS Statistics
└── plots/
    ├── handover_success_rate.png
    ├── false_handover_rate.png
    ├── handover_latency.png          (boxplot theo version, cần ≥1 trial evaluable)
    ├── source_switching.png
    ├── vps_localization_time.png
    └── gps_accuracy_distribution.png
```

Trial `INVALID` **không** bị âm thầm loại bỏ — vẫn xuất hiện trong
`per_trial_metrics.csv`/`validation_report.csv` (tô đỏ trong Excel), chỉ bị
loại khỏi `aggregate_results.csv`/plots nếu bật `--exclude-invalid`.

## 8. Cấu trúc code

```
harmony_analysis/
├── analyze_experiments.py   # CLI
├── gui.py                   # Tkinter GUI (wrapper thuần, không có logic phân tích)
├── config.yaml              # MỌI rule/definition cấu hình được ở đây
├── harmony_analysis/
│   ├── discovery.py         # recursive scan tìm events/samples/summary CSV
│   ├── loader.py             # load CSV an toàn, bắt lỗi corrupted/empty/missing cột
│   ├── pairing.py            # ghép 3 file theo session_id ĐỌC TỪ BÊN TRONG CSV
│   ├── events.py             # taxonomy event, tái tạo source switch/VPS attempt/handover
│   ├── metrics.py            # tính toàn bộ KPI per-trial + cross-check summary
│   ├── validation.py         # validation_status/notes (VALID/WARNINGS/INVALID/NOT_EVALUABLE)
│   ├── aggregate.py          # group + thống kê (N/mean/median/SD/Q1/Q3/CI)
│   ├── export.py             # ghi CSV + Excel (freeze header, autofilter, highlight)
│   ├── plotting.py           # publication-friendly plots (matplotlib)
│   └── pipeline.py           # orchestration dùng chung cho CLI & GUI
└── tests/                    # 30 unit/integration tests (pytest)
```

## 9. Chạy test

```bash
pip install pytest
pytest tests/ -v
```

30/30 test pass, bao gồm:
- `test_pairing.py`: ghép trial theo session_id bất kể tên thư mục, phát hiện
  file trùng/thiếu.
- `test_events.py`: reconstruct source switch, VPS attempt (success &
  timeout-fallback), **handover GPS→VPS bị tính là FAILURE dù to_state =
  IndoorVps** (regression chính của cả tool), false handover, oscillation.
- `test_validation.py`: CSV rỗng → INVALID; elapsed_s non-monotonic →
  warning; session_id mismatch giữa các file.
- `test_metrics_integration.py`: chạy full metrics trên đúng 3 file mẫu thật
  bạn cung cấp, khẳng định `handover_success_reconstructed == 0`,
  `source_transitions_total == 3`, không bịa `handover_latency_s`.
- `test_aggregate.py`: Wilson CI cho tỷ lệ, t-CI cho giá trị liên tục.
- `test_pipeline.py`: chạy end-to-end, kiểm tra đủ 6 file output + thư mục plots.

## 10. Đã chạy thử trên dữ liệu thật

- **1 trial mẫu** (3 file bạn cung cấp): xem `analysis_output/` đính kèm.
  `validation_status = VALID_WITH_WARNINGS` (cảnh báo
  `[TRIAL_NOT_COMPLETED]` vì `summary.completed=0` và không có
  `end_reason` — đúng thực tế vì VPS timeout khiến trial không tới đích).
- **Stress test 300 trial giả lập** (nhân bản có đổi `session_id`, rải ngẫu
  nhiên vào 5 version × 3 scenario folder khác nhau): chạy xong toàn bộ pipeline
  (scan → pair → metrics → aggregate → CSV → Excel → plots) trong **~11 giây**,
  và `harmony_version`/`direction` trong kết quả **lấy đúng từ nội dung CSV**
  (không lấy nhầm từ tên thư mục ngẫu nhiên) — xác nhận yêu cầu "không phụ
  thuộc cứng vào tên folder".

## 11. Giới hạn đã biết (ghi thẳng, không giấu)

- Chỉ có 1 trial mẫu thật (hướng GPS→VPS, kết quả FAIL) để verify logic khi
  bạn giao việc này — **chưa có ca VPS→GPS thật** hay **ca GPS→VPS SUCCESS
  thật** nào để kiểm chứng nhánh "success" của `detect_handover()`. Nhánh đó
  được viết theo đúng tên event/state trong taxonomy nhưng **nên chạy lại
  trên vài trial thật thành công** trước khi dùng số cho paper.
- `position_jump_m`/`heading_jump_deg` reconstructed dựa trên
  `campus_x/campus_y` quanh ±2s — ngưỡng 2s là giả định hợp lý nhưng **chưa
  được bạn xác nhận**; chỉnh trong `metrics.py::_reconstruct_position_jump`
  nếu cần cửa sổ khác.
- `revert_window_s=8.0` và `oscillation.window_s=15.0` là giá trị khởi điểm
  hợp lý dựa trên nhịp độ log quan sát được, **cấu hình được trong
  `config.yaml`**, không phải hằng số cứng — bạn nên tinh chỉnh khi có nhiều
  trial hơn.
