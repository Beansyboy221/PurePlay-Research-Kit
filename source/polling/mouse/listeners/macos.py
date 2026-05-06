import Quartz

from . import base


class RawMouseListener(base.BaseMouseListener):
    def __init__(self):
        super().__init__()
        self._run_loop = None
        self._last_pos = None

    def _event_callback(self, proxy, type_, event, refcon):
        if type_ in (
            Quartz.kCGEventMouseMoved,
            Quartz.kCGEventLeftMouseDragged,
            Quartz.kCGEventRightMouseDragged,
            Quartz.kCGEventOtherMouseDragged,
        ):
            pos = Quartz.CGEventGetLocation(event)
            with self._lock:
                if self._last_pos:
                    self._delta_x += pos.x - self._last_pos[0]
                    self._delta_y += pos.y - self._last_pos[1]
                self._last_pos = (pos.x, pos.y)
        return event

    def _run(self):
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGHIDEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            Quartz.kCGEventTapOptionDefault,
            Quartz.kCGEventMaskForAllEvents,
            self._event_callback,
            None,
        )
        if not tap:
            return
        Quartz.CGEventTapEnable(tap, True)
        self._run_loop = Quartz.CFRunLoopGetCurrent()
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(self._run_loop, source, Quartz.kCFRunLoopCommonModes)
        self.running = True
        Quartz.CFRunLoopRun()
        self.running = False

    def stop(self):
        if self._run_loop:
            Quartz.CFRunLoopStop(self._run_loop)
        if self._thread:
            self._thread.join(timeout=1.0)
