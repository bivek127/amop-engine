import pytest

from exceptions import TaskNotFoundError
from models import Task
from storage import InMemoryTaskStore


def make_task(task_id="1"):
    return Task(id=task_id, title="x", description="", urgency=5, impact=5, effort=5)


def test_add_and_get():
    store = InMemoryTaskStore()
    store.add(make_task())
    assert store.get("1").title == "x"


def test_get_missing_raises():
    store = InMemoryTaskStore()
    with pytest.raises(TaskNotFoundError):
        store.get("missing")


def test_update_missing_raises():
    store = InMemoryTaskStore()
    with pytest.raises(TaskNotFoundError):
        store.update(make_task())


def test_delete_and_count():
    store = InMemoryTaskStore()
    store.add(make_task())
    assert store.count() == 1
    store.delete("1")
    assert store.count() == 0


def test_list_returns_all():
    store = InMemoryTaskStore()
    store.add(make_task("1"))
    store.add(make_task("2"))
    assert {t.id for t in store.list()} == {"1", "2"}
