import queue
import threading


class SpeechWorker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._backend = None
        self._failed = False

        if not enabled:
            return

        try:
            import pyttsx3  # noqa: F401

            self._backend = "pyttsx3"
        except Exception:
            try:
                import winsound  # noqa: F401

                self._backend = "winsound"
            except Exception:
                self._failed = True
                return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        engine = None
        winsound_module = None

        try:
            if self._backend == "pyttsx3":
                import pyttsx3

                engine = pyttsx3.init()
            elif self._backend == "winsound":
                import winsound

                winsound_module = winsound
        except Exception:
            self._failed = True
            return

        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break

            try:
                if self._backend == "pyttsx3":
                    engine.say(text)
                    engine.runAndWait()
                elif self._backend == "winsound":
                    print(f"[audio] {text}")
                    winsound_module.MessageBeep()
            except Exception:
                self._failed = True
            finally:
                self._queue.task_done()

        if engine is not None:
            engine.stop()

    def say(self, text: str) -> None:
        if self.enabled and not self._failed and text:
            self._queue.put(text)

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=5)
