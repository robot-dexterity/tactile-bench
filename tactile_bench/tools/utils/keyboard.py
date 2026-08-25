"""Keyboard input utilities for interactive robot tests."""

from pynput import keyboard


class Keyboard:
    """Track the keys currently held down by the user."""

    def __init__(self) -> None:
        self.pressed = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)

    def _on_press(self, key) -> None:
        self.pressed.add(key)

    def _on_release(self, key) -> None:
        self.pressed.discard(key)

    def __enter__(self) -> "Keyboard":
        self.listener.start()
        return self

    def __exit__(self, *_) -> None:
        self.listener.stop()

    def get_state(self) -> frozenset:
        """Return a snapshot of the keys currently held down."""
        return frozenset(self.pressed)
