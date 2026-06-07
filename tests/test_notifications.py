"""Tests for core.notifications module"""
import pytest
import tempfile
import os
from core.notifications import NotificationManager, Notification


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def nm(temp_dir):
    path = os.path.join(temp_dir, "test_notifications.json")
    return NotificationManager(storage_path=path)


class TestNotificationInit:
    def test_notification_creation(self):
        n = Notification("user1", "Title", "Message", "info")
        assert n.user_id == "user1"
        assert n.title == "Title"
        assert n.message == "Message"
        assert n.type == "info"
        assert n.read is False
        assert n.id is not None

    def test_notification_to_dict(self):
        n = Notification("user1", "Title", "Message", "warning")
        d = n.to_dict()
        assert d["user_id"] == "user1"
        assert d["title"] == "Title"
        assert d["type"] == "warning"
        assert d["read"] is False


class TestNotificationManagerInit:
    def test_init_creates_manager(self, nm):
        assert nm is not None
        assert isinstance(nm.notifications, list)

    def test_init_empty(self, nm):
        assert len(nm.notifications) == 0


class TestAddNotification:
    def test_add_single(self, nm):
        result = nm.add_notification("user1", "Title", "Message", "info")
        assert result is True
        assert len(nm.notifications) == 1

    def test_add_multiple(self, nm):
        nm.add_notification("user1", "Title 1", "Msg 1", "info")
        nm.add_notification("user1", "Title 2", "Msg 2", "warning")
        nm.add_notification("user2", "Title 3", "Msg 3", "error")
        assert len(nm.notifications) == 3

    def test_add_persists(self, temp_dir):
        path = os.path.join(temp_dir, "persist.json")
        nm1 = NotificationManager(storage_path=path)
        nm1.add_notification("user1", "Title", "Message", "info")

        nm2 = NotificationManager(storage_path=path)
        assert len(nm2.notifications) == 1


class TestGetNotifications:
    def test_get_all(self, nm):
        nm.add_notification("user1", "Title 1", "Msg 1", "info")
        nm.add_notification("user1", "Title 2", "Msg 2", "warning")
        nm.add_notification("user2", "Title 3", "Msg 3", "error")

        result = nm.get_notifications("user1")
        assert len(result) == 2

    def test_get_unread_only(self, nm):
        nm.add_notification("user1", "Title 1", "Msg 1", "info")
        nm.add_notification("user1", "Title 2", "Msg 2", "warning")

        notifs = nm.get_notifications("user1")
        nm.mark_read(notifs[0]["id"])

        result = nm.get_notifications("user1", unread_only=True)
        assert len(result) == 1

    def test_get_empty(self, nm):
        result = nm.get_notifications("nonexistent")
        assert len(result) == 0


class TestMarkRead:
    def test_mark_read_success(self, nm):
        nm.add_notification("user1", "Title", "Message", "info")
        notifs = nm.get_notifications("user1")
        result = nm.mark_read(notifs[0]["id"])
        assert result is True

    def test_mark_read_updates(self, nm):
        nm.add_notification("user1", "Title", "Message", "info")
        notifs = nm.get_notifications("user1")
        nm.mark_read(notifs[0]["id"])

        unread = nm.get_notifications("user1", unread_only=True)
        assert len(unread) == 0

    def test_mark_read_nonexistent(self, nm):
        result = nm.mark_read("nonexistent_id")
        assert result is False


class TestGetUnreadCount:
    def test_count_empty(self, nm):
        assert nm.get_unread_count("user1") == 0

    def test_count_all_unread(self, nm):
        nm.add_notification("user1", "Title 1", "Msg 1", "info")
        nm.add_notification("user1", "Title 2", "Msg 2", "warning")
        assert nm.get_unread_count("user1") == 2

    def test_count_partial_read(self, nm):
        nm.add_notification("user1", "Title 1", "Msg 1", "info")
        nm.add_notification("user1", "Title 2", "Msg 2", "warning")
        notifs = nm.get_notifications("user1")
        nm.mark_read(notifs[0]["id"])
        assert nm.get_unread_count("user1") == 1


class TestDeleteNotification:
    def test_delete_existing(self, nm):
        nm.add_notification("user1", "Title", "Message", "info")
        notifs = nm.get_notifications("user1")
        result = nm.delete_notification(notifs[0]["id"])
        assert result is True
        assert len(nm.notifications) == 0

    def test_delete_nonexistent(self, nm):
        result = nm.delete_notification("nonexistent")
        assert result is False


class TestClearRead:
    def test_clear_read(self, nm):
        nm.add_notification("user1", "Title 1", "Msg 1", "info")
        nm.add_notification("user1", "Title 2", "Msg 2", "warning")
        notifs = nm.get_notifications("user1")
        nm.mark_read(notifs[0]["id"])

        cleared = nm.clear_read("user1")
        assert cleared == 1
        assert len(nm.get_notifications("user1")) == 1
