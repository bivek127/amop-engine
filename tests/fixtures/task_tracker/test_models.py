from models import Status, Task


def test_task_defaults_to_open():
    task = Task(id="1", title="x", description="", urgency=5, impact=5, effort=5)
    assert task.status == Status.OPEN
    assert task.assignee is None
    assert task.tags == []


def test_is_terminal():
    task = Task(id="1", title="x", description="", urgency=5, impact=5, effort=5, status=Status.DONE)
    assert task.is_terminal()

    task.status = Status.OPEN
    assert not task.is_terminal()
