import asyncio

import pytest

from backend.app import main as main_module


@pytest.mark.parametrize("path", ["student", "teacher", "index.html"])
def test_spa_entry_responses_disable_browser_cache(tmp_path, monkeypatch, path):
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
    monkeypatch.setattr(main_module, "frontend_dist", frontend_dist)

    response = asyncio.run(main_module.frontend_app(path))

    assert response.headers["cache-control"] == (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["expires"] == "0"


def test_static_file_response_keeps_normal_cache_policy(tmp_path, monkeypatch):
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
    (frontend_dist / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main_module, "frontend_dist", frontend_dist)

    response = asyncio.run(main_module.frontend_app("manifest.json"))

    assert "cache-control" not in response.headers
