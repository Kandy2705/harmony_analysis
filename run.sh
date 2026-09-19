#!/usr/bin/env bash
# ==============================================================================
# HARMONY Analysis One-Click Pipeline Runner
# ==============================================================================
set -e

# Change to the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Select Python environment
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

DATA_DIR="${1:-sample_data}"
OUTPUT_DIR="${2:-./analysis_output}"

echo "============================================================"
echo "🚀 HARMONY Analysis Pipeline"
echo "• Python  : $PYTHON"
echo "• Data Dir: $DATA_DIR"
echo "• Output  : $OUTPUT_DIR"
echo "============================================================"

"$PYTHON" analyze_experiments.py "$DATA_DIR" --output "$OUTPUT_DIR" --exclude-invalid

echo ""
echo "✅ Phân tích hoàn tất! Kết quả đã được lưu tại: $OUTPUT_DIR"
echo "• Bảng số liệu Paper : $OUTPUT_DIR/paper_metrics.csv"
echo "• Bảng chi tiết      : $OUTPUT_DIR/aggregate_results.csv"
echo "• Biểu đồ (Plots)    : $OUTPUT_DIR/plots/"
echo "============================================================"
