import logging
import time
from threading import Lock, Thread


class ProgramLearning:
    def __init__(self, build):
        self.build = build
        self.lock = Lock()
        self.events = []
        self.running = False
        self.error = ""
        self.run = 1
        self.bundle = build(self.record)

    def record(self, phase, item, detail):
        with self.lock:
            self.events.append({"phase": phase, "item": item, "detail": detail})
            live = self.running
        if live:
            time.sleep(0.012)

    def start(self):
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.error = ""
            self.events = []
            self.run += 1
        Thread(target=self.retrain, daemon=True).start()
        return True

    def retrain(self):
        try:
            bundle = self.build(self.record)
            with self.lock:
                self.bundle = bundle
        except Exception:
            logging.exception("Program training failed")
            with self.lock:
                self.error = "학습에 실패했습니다. 다시 시작할 수 있습니다."
        finally:
            with self.lock:
                self.running = False

    def snapshot(self):
        with self.lock:
            return {"run": self.run, "running": self.running, "error": self.error, "events": self.events.copy()}
