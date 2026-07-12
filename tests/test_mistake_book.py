import asyncio

from backend.app.services.mistake_book import MistakeBook


def test_mistake_book_is_durable_deduplicated_and_student_scoped(tmp_path):
    path = tmp_path / "mistakes.json"
    book = MistakeBook(path)
    first = asyncio.run(book.add(
        student_id="learner-a",
        session_id="student-session-a",
        content="求二极管恒压降模型下的回路电流。",
        agent="答疑 Agent",
        knowledge_points=["二极管", "恒压降模型", "二极管"],
        summary="二极管恒压降计算",
    ))
    duplicate = asyncio.run(book.add(
        student_id="learner-a",
        session_id="student-session-b",
        content="求二极管恒压降模型下的回路电流。",
        agent="出题 Agent",
        knowledge_points=["二极管"],
        summary="重复题",
    ))
    asyncio.run(book.add(
        student_id="learner-b",
        session_id="student-session-c",
        content="分析晶体管静态工作点。",
        agent="答疑 Agent",
        knowledge_points=["晶体管", "静态工作点"],
        summary="静态工作点",
    ))

    assert first["id"] == duplicate["id"]
    assert first["knowledge_points"] == ["二极管", "恒压降模型"]
    assert len(asyncio.run(MistakeBook(path).list("learner-a"))) == 1
    assert len(asyncio.run(MistakeBook(path).list("learner-b"))) == 1
    assert asyncio.run(book.delete("learner-a", first["id"])) is True
    assert asyncio.run(book.list("learner-a")) == []
