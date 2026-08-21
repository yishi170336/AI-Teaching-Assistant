from __future__ import annotations

from backend.app.rag.paddleocr_vl import (
    PADDLEOCR_VL_GIT_REVISION,
    markdown_table_cells,
    normalize_paddle_result,
    paddle_result_to_dict,
)


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
