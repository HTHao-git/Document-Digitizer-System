# Module 2 - OCR Engine (Vintern)

## Cấu trúc
```
module2-ocr/
├── data/
│   └── bbox/           # ouput vẽ Bounding Box
│   └── json/           # JSON output chứa text & toạ độ
├── schemas/
│   ├── config.schema.json # schema validate config
│   └── ocr.schema.json    # schema validate output
├── scripts/
│   ├── draw_bbox.py    # script hỗ trợ QA vẽ toạ độ lên ảnh
│   └── patch_craft_text_detector.py # patch CRAFT lỗi numpy jagged polygons (dtype=object)
├── src/
│   ├── engines/
│   │   └── vintern_engine.py # VLM engine xử lý đọc chữ và tính điểm
│   ├── config.py       # tham số
│   ├── logger.py       # helper log
│   ├── ocr.py          # pipeline chính 
│   ├── run.py          # CLI entry point
│   └── schema.py       # helper validate
├── tests/
│   ├── conftest.py
│   └── test_contract.py
│   └── test_ocr.py      # test chính cho ocr.py (blank page + QA flags)
│   └── test_qa_flags_thresholds.py      # test Business Logic UI Các trường có độ tin cậy thấp.
├── config.yaml         # file cấu hình mặc định
├── requirements.txt    # danh sách thư viện
└── README.md
```

## 1. Run
*Lưu ý: Bắt buộc dùng Python 3.9 và Virtual Environment. Tránh Python 3.13 vì `numpy==1.26.4` không có wheel và rủi ro Dependency Hell.*

> Không dùng Virtual Environment (`venv` hoặc Conda) có thể gây ra lỗi *Dependency Hell*, xung đột thư viện với các dự án khác, làm hỏng driver của máy, và gây khó khăn khi triển khai lên server Production.

```bash
# Tạo và kích hoạt môi trường ảo (Windows PowerShell)
py -3.9 -m venv .venv39
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv39\Scripts\Activate.ps1

# Cài đặt thư viện
python -m pip install --upgrade pip
python -m pip install opencv-python-headless==4.9.0.80 numpy==1.26.4
python -m pip install -r requirements.txt

# Lưu ý: `requirements.txt` không include torch để tránh pip tự kéo nhầm bản (CPU/CUDA)
# Cài torch theo 1 trong 2 cách bên dưới (CPU hoặc CUDA) trước khi chạy OCR.

# Cài torch GPU (CUDA 12.1) - khuyến nghị
# Lưu ý: wheel ~2.5GB, cần đủ dung lượng trống 
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài torch CPU (nhẹ hơn)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Nếu cần Bounding Boxes (Hybrid): cài thêm CRAFT 
python -m pip install craft-text-detector --no-deps
python -m pip install shapely scikit-image gdown

# Chạy OCR pipeline
python scripts/patch_craft_text_detector.py
python -m src.run --input-json ./../preprocess/data/json --output-json ./data/json --config config.yaml
```
Lưu ý: Lần chạy đầu có thể mất nhiều thời gian do tải weights (CRAFT/Vintern) và load model. CLI sẽ log tiến trình theo từng file vào console và file log (nếu dùng `--log`).

## Visual QA (Bounding Boxes)
Vẽ bounding box (input là folder JSON, output là folder ảnh):
```bash
python scripts/draw_bbox.py --json data/json --output data/bbox --no-text
```

## 2. Config
Tất cả params được định nghĩa trong `config.yaml`.  
Config được validate thông qua `schemas/config.schema.json`. 

## 3. Output JSON
Mỗi file JSON từ Module 1 sẽ tạo ra một file JSON tương ứng trong `output-json` folder.

**Output schema:** `schemas/ocr.schema.json`

**Important fields:**
- `request_id`: UUID 
- `document_id`: string
- `status`: success | error 
- `payload.page`: page index 
- `payload.text`: Raw text đã được trích xuất 
- `payload.blocks`: Mảng chứa từng dòng, toạ độ (`bbox`) và điểm tự tin (`confidence`). Nếu `behavior.enable_bounding_boxes=false` thì mảng này rỗng.
- `meta.overall_confidence`: Điểm tin cậy trung bình của cả trang
- `meta.qa_flags`: Tổng hợp các dòng/trang có độ tin cậy thấp để UI highlight QA (vàng/đỏ)

Sau mỗi lần chạy batch, hệ thống tạo file `output-json/_summary.json` chứa thống kê tổng:
- `average_overall_confidence`: trung bình confidence của tất cả trang thành công (có `overall_confidence`)
- `qa_flagged_pages`, `qa_flagged_blocks`, `qa_flagged_level_counts`

## 4. Testing
```bash
python -m pip install pytest
pytest -q
```

## 5. Config Reference

### engine
- `name` (string): `vintern`.
- `language` (string): `vie` (Tiếng Việt).

### Engine-specific params (Vintern)
- `model_name` (string): Tên model trên Hugging Face (Mặc định: `5CD-AI/Vintern-1B-v3_5`).
- `device` (string): `auto` / `cpu` / `cuda`. 
- `trust_remote_code` (bool): Cần thiết để tải custom model kiến trúc InternVL.
- `max_new_tokens` (int): Giới hạn số lượng token sinh ra tối đa.
- `temperature` (float): Nhiệt độ lấy mẫu (Mặc định 0.0 để đọc chữ chính xác nhất, không sáng tạo).

### behavior
- `skip_blank_pages` (bool): Bỏ qua không chạy OCR nếu Module 1 đánh dấu `is_blank = true`.
- `detector` (string): `craft` | `cv`. Chọn engine detect bounding boxes.
- `join_lines` (bool): Ghép các dòng liền nhau thành đoạn trong `payload.text` .
- `bbox_pad_x_ratio` (number): Tỷ lệ nới bbox theo chiều ngang khi crop để OCR .
- `bbox_pad_y_ratio` (number): Tỷ lệ nới bbox theo chiều dọc khi crop để OCR.
- `bbox_pad_px` (int): Padding cố định (pixel) cộng thêm vào bbox khi crop.
- `min_ink_ratio` (number, 0-1): Bỏ qua bbox quá “trắng” (ít chữ) để tiết kiệm thời gian OCR.
- `vintern_crop_max_num` (int): Số tile tối đa khi Vintern xử lý ảnh crop.
- `vintern_crop_max_new_tokens` (int): Giới hạn token tối đa cho mỗi crop.
- `confidence_threshold_yellow` (number, 0-100): Ngưỡng dưới mức này sẽ bị flag vàng để QA1 chú ý.
- `confidence_threshold_red` (number, 0-100): Ngưỡng dưới mức này sẽ bị flag đỏ để QA1 cảnh báo mạnh hơn.
