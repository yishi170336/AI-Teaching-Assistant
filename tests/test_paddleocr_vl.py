from __future__ import annotations

import io
import json

from PIL import Image

from backend.app.rag.paddleocr_vl import (
    PADDLEOCR_VL_GIT_REVISION,
    PaddleOCRVLAPIClient,
    PaddleOCRVLConfig,
    markdown_table_cells,
    normalize_paddle_result,
    paddle_result_to_dict,
)


class _FakeResponse:
    def __init__(self, *, payload=None, text="", status_code=200):
        self._payload = payload
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not 200 <= self.status_code < 300:
            raise RuntimeError(f"HTTP {self.status_code}")


def _tiny_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 200), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _result(blocks, boxes):
    return {
        "width": 1000,
        "height": 1000,
        "parsing_res_list": blocks,
        "layout_det_res": {"boxes": boxes},
    }


def test_live_result_json_wrapper_is_unwrapped():
    class Result:
        json = {"res": {"width": 100, "parsing_res_list": []}}

    assert paddle_result_to_dict(Result()) == {
        "width": 100,
        "parsing_res_list": [],
    }


def test_normalize_paddle_result_repairs_same_column_vertical_inversion():
    result = _result(
        [
            {
                "block_id": 8,
                "block_label": "paragraph_title",
                "block_content": "1.1 半导体",
                "block_bbox": [100, 400, 900, 450],
                "block_order": 1,
            },
            {
                "block_id": 7,
                "block_label": "text",
                "block_content": "前一段正文应先阅读。",
                "block_bbox": [100, 300, 900, 360],
                "block_order": 2,
            },
        ],
        [
            {"label": "paragraph_title", "coordinate": [100, 400, 900, 450], "score": 0.94},
            {"label": "text", "coordinate": [100, 300, 900, 360], "score": 0.96},
        ],
    )

    blocks = normalize_paddle_result(result, source="lesson.pdf", page=3)

    assert [block["text"] for block in blocks] == [
        "前一段正文应先阅读。",
        "1.1 半导体",
    ]
    assert blocks[1]["type"] == "section_heading"
    assert blocks[1]["corrections"][0]["type"] == "geometry-reading-order"
    assert blocks[0]["id"] == "ocr:lesson.pdf:p3:b7"
    assert blocks[0]["model_revision"] == PADDLEOCR_VL_GIT_REVISION


def test_reading_order_preserves_paddle_order_across_columns():
    result = _result(
        [
            {
                "block_id": 1,
                "block_label": "text",
                "block_content": "左栏后段",
                "block_bbox": [40, 700, 450, 780],
                "block_order": 1,
            },
            {
                "block_id": 2,
                "block_label": "text",
                "block_content": "右栏前段",
                "block_bbox": [550, 100, 960, 180],
                "block_order": 2,
            },
        ],
        [
            {"label": "text", "coordinate": [40, 700, 450, 780], "score": 0.93},
            {"label": "text", "coordinate": [550, 100, 960, 180], "score": 0.95},
        ],
    )

    blocks = normalize_paddle_result(result, source="columns.pdf", page=1)

    assert [block["text"] for block in blocks] == ["左栏后段", "右栏前段"]
    assert all(not block["corrections"] for block in blocks)


def test_list_items_are_recovered_from_aligned_neighbours():
    result = _result(
        [
            {
                "block_id": 1,
                "block_label": "text",
                "block_content": "1. 输入电阻高",
                "block_bbox": [120, 100, 700, 150],
                "block_order": 1,
            },
            {
                "block_id": 2,
                "block_label": "text",
                "block_content": "输出电阻低",
                "block_bbox": [122, 170, 700, 220],
                "block_order": 2,
            },
        ],
        [
            {"label": "text", "coordinate": [120, 100, 700, 150], "score": 0.94},
            {"label": "text", "coordinate": [122, 170, 700, 220], "score": 0.94},
        ],
    )

    blocks = normalize_paddle_result(result, source="list.pdf", page=1)

    assert [block["type"] for block in blocks] == ["list_item", "list_item"]
    assert blocks[1]["corrections"][0]["type"] == "geometry-list-recovery"


def test_markdown_table_cells_preserve_headers_and_cell_ids():
    cells = markdown_table_cells(
        "| 参数 | 典型值 | 单位 |\n|---|---:|---|\n| 电压增益 | 40 | dB |",
        block_id="ocr:lesson.pdf:p5:b9",
    )

    gain = next(cell for cell in cells if cell["column_header"] == "典型值")
    assert gain == {
        "id": "ocr:lesson.pdf:p5:b9:r1:c1",
        "row": 1,
        "column": 1,
        "row_header": "电压增益",
        "column_header": "典型值",
        "value": "40",
    }


def test_api_client_normalizes_pruned_result_without_leaking_token(monkeypatch):
    token = "local-test-secret"
    status_calls = 0

    class FakeSession:
        def __init__(self):
            self.post_headers = {}
            self.get_headers = []

        def post(self, _url, *, headers, data, files, timeout):
            self.post_headers = dict(headers)
            assert data["model"] == "PaddleOCR-VL-1.6"
            assert files["file"][2] == "image/png"
            assert timeout > 0
            return _FakeResponse(payload={"data": {"jobId": "job-1"}})

        def get(self, _url, *, headers, timeout):
            nonlocal status_calls
            status_calls += 1
            self.get_headers.append(dict(headers))
            if status_calls == 1:
                return _FakeResponse(payload={"data": {"state": "running"}})
            return _FakeResponse(payload={
                "data": {
                    "state": "done",
                    "resultUrl": {"jsonUrl": "https://objects.example/result.jsonl"},
                }
            })

        def close(self):
            pass

    pruned = _result(
        [{
            "block_id": 1,
            "block_label": "formula",
            "block_content": r"$I_C=\beta I_B$",
            "block_bbox": [10, 20, 90, 60],
            "block_order": 1,
        }],
        [{"label": "formula", "coordinate": [10, 20, 90, 60], "score": 0.95}],
    )
    jsonl = json.dumps({
        "result": {
            "layoutParsingResults": [{
                "prunedResult": pruned,
                "markdown": {"text": r"$I_C=\beta I_B$", "images": {}},
            }]
        }
    })
    result_headers = []

    def fetch_result(_url, *, timeout):
        result_headers.append(None)
        assert timeout > 0
        return _FakeResponse(text=jsonl)

    monkeypatch.setattr("backend.app.rag.paddleocr_vl.requests.get", fetch_result)
    session = FakeSession()
    client = PaddleOCRVLAPIClient(
        token=token,
        poll_interval_seconds=0,
        session=session,
    )

    result = client.predict_page(_tiny_png(), source="lesson.pdf", page=7)

    assert result["blocks"][0]["type"] == "formula"
    assert result["blocks"][0]["source_engine"] == "paddleocr-vl-api"
    assert result["blocks"][0]["model_revision"] == "aistudio-api:PaddleOCR-VL-1.6"
    assert session.post_headers == {"Authorization": f"bearer {token}"}
    assert all(headers == session.post_headers for headers in session.get_headers)
    assert result_headers == [None]
    assert token not in repr(PaddleOCRVLConfig(provider="api", api_token=token))
