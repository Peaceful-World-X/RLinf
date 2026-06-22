# Copyright 2026 The GIGA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading


class KeyboardListener:
    """Headless keyboard listener backed by Linux evdev input devices."""

    REQUIRED_KEY_NAMES = ("KEY_A", "KEY_B", "KEY_C", "KEY_D", "KEY_Q")

    def __init__(self):
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError as exc:
            raise RuntimeError(
                "KeyboardListener requires the 'evdev' package. "
                "Install the real-world extras with evdev support."
            ) from exc

        self._input_device_cls = InputDevice
        self._ecodes = ecodes
        self._list_devices = list_devices

        self.state_lock = threading.Lock()
        self.latest_data = {"key": None}
        self._pressed_keys: set[str] = set()
        self._pending_keys: list[str] = []
        self.device = self._open_keyboard_device()

        self.listener = threading.Thread(
            target=self._listen_loop,
            name=f"KeyboardListener:{self.device.path}",
            daemon=True,
        )
        self.listener.start()
        self.last_intervene = 0

    def _open_keyboard_device(self):
        override_path = os.environ.get("RLINF_KEYBOARD_DEVICE")
        if override_path:
            device = self._open_device(override_path, is_override=True)
            if not self._is_keyboard_device(device):
                device.close()
                raise RuntimeError(
                    "KeyboardListener device set by "
                    f"RLINF_KEYBOARD_DEVICE='{override_path}' does not look like a "
                    "keyboard device. Point it to the correct /dev/input/eventX path."
                )
            return device

        permission_denied_paths: list[str] = []
        for device_path in sorted(self._list_devices()):
            try:
                device = self._open_device(device_path)
            except PermissionError:
                permission_denied_paths.append(device_path)
                continue

            if self._is_keyboard_device(device):
                return device
            device.close()

        if permission_denied_paths:
            denied = ", ".join(permission_denied_paths)
            raise RuntimeError(
                "KeyboardListener could not open any readable keyboard device under "
                f"/dev/input/event*. Permission denied for: {denied}. Grant the runtime "
                "user read access via the input group or udev rules, or set "
                "RLINF_KEYBOARD_DEVICE to a readable keyboard event device."
            )

        raise RuntimeError(
            "KeyboardListener could not find a readable keyboard device under "
            "/dev/input/event*. Ensure a physical keyboard is connected, the runtime "
            "user has access to input devices, or set RLINF_KEYBOARD_DEVICE to the "
            "correct /dev/input/eventX path."
        )

    def _open_device(self, device_path: str, is_override: bool = False):
        try:
            return self._input_device_cls(device_path)
        except FileNotFoundError as exc:
            if is_override:
                raise RuntimeError(
                    f"KeyboardListener override path '{device_path}' does not exist."
                ) from exc
            raise
        except PermissionError as exc:
            if is_override:
                raise RuntimeError(
                    "KeyboardListener cannot read the device set by "
                    f"RLINF_KEYBOARD_DEVICE='{device_path}'. Grant the runtime user "
                    "read access via the input group or udev rules."
                ) from exc
            raise
        except OSError as exc:
            if is_override:
                raise RuntimeError(
                    "KeyboardListener failed to open the device set by "
                    f"RLINF_KEYBOARD_DEVICE='{device_path}': {exc}"
                ) from exc
            raise RuntimeError(
                f"KeyboardListener failed to open input device '{device_path}': {exc}"
            ) from exc

    def _is_keyboard_device(self, device) -> bool:
        required_codes = {
            getattr(self._ecodes, key_name) for key_name in self.REQUIRED_KEY_NAMES
        }
        capabilities = device.capabilities(verbose=False)
        supported_key_codes = set(capabilities.get(self._ecodes.EV_KEY, []))
        return required_codes.issubset(supported_key_codes)

    def _listen_loop(self) -> None:
        try:
            for event in self.device.read_loop():
                if event.type != self._ecodes.EV_KEY:
                    continue

                key = self._event_to_key(event.code)
                if key is None:
                    continue

                if event.value in (1, 2):
                    with self.state_lock:
                        self.latest_data["key"] = key
                        self._pressed_keys.add(key)
                        if event.value == 1:
                            self._pending_keys.append(key)
                elif event.value == 0:
                    with self.state_lock:
                        self._pressed_keys.discard(key)
                        if self.latest_data["key"] == key:
                            self.latest_data["key"] = None
        finally:
            with self.state_lock:
                self.latest_data["key"] = None
                self._pressed_keys.clear()
                self._pending_keys.clear()
            self.device.close()

    def _event_to_key(self, key_code: int) -> str | None:
        key_name = self._ecodes.bytype[self._ecodes.EV_KEY].get(key_code)
        if isinstance(key_name, list):
            key_name = key_name[0]
        if not isinstance(key_name, str):
            return None

        if key_name.startswith("KEY_"):
            normalized_key = key_name.removeprefix("KEY_").lower()
            if len(normalized_key) == 1:
                return normalized_key
            return f"Key.{normalized_key}"
        return key_name.lower()

    def get_key(self) -> str | None:
        """Returns the currently held key."""
        with self.state_lock:
            return self.latest_data["key"]

    def consume_press(self, key: str) -> bool:
        """Returns True if key was pressed since last call, then clears it.

        Use this instead of get_key() for toggle keys to avoid missing
        short presses between polling cycles.
        """
        with self.state_lock:
            if key in self._pending_keys:
                self._pending_keys.remove(key)
                return True
            if key in self._pressed_keys:
                self._pressed_keys.discard(key)
                return True
            return False

    def consume_any_press(
        self,
        keys: tuple[str, ...] | list[str] | set[str],
        *,
        include_held: bool = True,
    ) -> str | None:
        """Returns and clears the oldest pending press among ``keys``.

        Falls back to currently-held keys when ``include_held`` is true.
        """
        key_set = set(keys)
        with self.state_lock:
            for idx, pending_key in enumerate(self._pending_keys):
                if pending_key in key_set:
                    return self._pending_keys.pop(idx)
            if not include_held:
                return None
            for pressed_key in tuple(self._pressed_keys):
                if pressed_key in key_set:
                    self._pressed_keys.discard(pressed_key)
                    return pressed_key
            return None

    def clear_keys(
        self, keys: tuple[str, ...] | list[str] | set[str] | None = None
    ) -> None:
        """Clear pending and currently-held keys.

        ``keys=None`` clears all cached key state. Passing a key collection clears
        only those keys.
        """
        with self.state_lock:
            if keys is None:
                self.latest_data["key"] = None
                self._pressed_keys.clear()
                self._pending_keys.clear()
                return
            key_set = set(keys)
            self._pressed_keys.difference_update(key_set)
            self._pending_keys = [
                pending_key
                for pending_key in self._pending_keys
                if pending_key not in key_set
            ]
            if self.latest_data["key"] in key_set:
                self.latest_data["key"] = None
