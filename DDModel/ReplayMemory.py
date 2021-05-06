import gc
import time
import redis
import threading

from _pickle import loads
from DDModel.Config import DDModelConfig


class Replay(threading.Thread):
    def __init__(
        self, Config: DDModelConfig
    ):
        super(Replay, self).__init__()
        self.setDaemon(True)
        self._cfg = Config
        self._lock = threading.Lock()
        self._connect = redis.StrictRedis("localhost")
        data = self._connect.scan()
        if data[-1] != []:
            self._connect.delete(*data[-1])
        self._buffer = []

        print(__name__)
        print("hello")

    def run(self):
        while True:
            pipe = self._connect.pipeline()
            pipe.lrange("data", 0, -1)
            pipe.ltrim("data", -1, 0)
            data = pipe.execute()[0]
            scan = self._connect.scan()
            if scan[-1] != []:
                self._connect.delete(*scan[-1])
            if data is not None:
            # with self._lock:
                for d in data:
                    x = loads(d)
                    self._buffer.insert(0, x)
            gc.collect()
            time.sleep(2.5)

    def sample(self):
        while len(self._buffer) < 10:
            time.sleep(1)
        return self._buffer.pop()

    def __len__(self):
        return len(self._name)
