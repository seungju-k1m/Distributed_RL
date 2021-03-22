import gc
import time
import threading
import redis

from baseline.utils import loads, ReplayMemory
from IMPALA.Config import IMPALAConfig


class Replay(threading.Thread):
    def __init__(
        self, config: IMPALAConfig, connect=redis.StrictRedis(host="localhost")
    ):
        super(Replay, self).__init__()

        # main thread가 종료되면 그 즉시 종료되는 thread이다.
        self.setDaemon(True)

        self._memory = ReplayMemory(int(config.replayMemory))

        self._connect = connect
        self._lock = threading.Lock()

    def run(self):
        t = 0
        while True:
            t += 1
            pipe = self._connect.pipeline()
            pipe.lrange("trajectory", 0, -1)
            pipe.ltrim("trajectory", -1, 0)
            data = pipe.execute()[0]
            if data is not None:
                for d in data:
                    with self._lock:
                        self._memory.push(d)
            # time.sleep(0.01)
            gc.collect()
            # if (t % 100) == 0:

    def sample(self, batch_size):
        with self._lock:
            return self._memory.sample(batch_size)

    def __len__(self):
        return len(self._memory)
