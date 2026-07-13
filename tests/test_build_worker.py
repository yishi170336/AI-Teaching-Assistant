import json

from backend.app.rag import build_worker


def test_worker_json_retries_windows_replace_lock(tmp_path, monkeypatch):
    target = tmp_path / ".build-progress.json"
    real_replace = build_worker.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "file is temporarily locked", destination)
        real_replace(source, destination)

    monkeypatch.setattr(build_worker.os, "replace", flaky_replace)
    monkeypatch.setattr(build_worker.time, "sleep", lambda _seconds: None)

    build_worker._write_json(target, {"progress": 42, "stage": "embedding"})

    assert attempts == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "progress": 42,
        "stage": "embedding",
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_best_effort_progress_write_does_not_abort_build(tmp_path, monkeypatch):
    target = tmp_path / ".build-progress.json"

    def locked_write(path, data, encoding=None):
        del encoding
        if path.suffix == ".tmp":
            return len(data)
        raise PermissionError(13, "locked", path)

    monkeypatch.setattr(
        build_worker.os,
        "replace",
        lambda _source, destination: (_ for _ in ()).throw(
            PermissionError(13, "file remains locked", destination)
        ),
    )
    monkeypatch.setattr(build_worker.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(build_worker.Path, "write_text", locked_write)

    build_worker._write_json(target, {"progress": 50}, best_effort=True)


def test_best_effort_progress_write_ignores_temp_file_lock(tmp_path, monkeypatch):
    target = tmp_path / ".build-progress.json"

    def locked_temp_write(_path, _data, encoding=None):
        del encoding
        raise PermissionError(13, "temporary file is locked")

    monkeypatch.setattr(build_worker.Path, "write_text", locked_temp_write)

    build_worker._write_json(target, {"progress": 60}, best_effort=True)


def test_temp_cleanup_lock_does_not_hide_success(tmp_path, monkeypatch):
    target = tmp_path / ".build-progress.json"
    real_unlink = build_worker.Path.unlink

    def locked_temp_unlink(path, missing_ok=False):
        if path.suffix == ".tmp":
            raise PermissionError(13, "temporary file cleanup is locked", path)
        return real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(build_worker.Path, "unlink", locked_temp_unlink)

    build_worker._write_json(target, {"progress": 70})

    assert json.loads(target.read_text(encoding="utf-8")) == {"progress": 70}
