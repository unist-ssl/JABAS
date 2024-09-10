import time
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    def __init__(self, msg=None):
        self.msg = msg
        self.elapsed = 0

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = time.time() - self.time