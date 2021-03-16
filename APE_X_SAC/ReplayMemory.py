import gc
import time
import threading
import redis

from baseline.utils import PrioritizedMemory, loads
from SAC.Config import SACConfig


class Replay(threading.Thread):
    def __init__(self, config: SACConfig, connect=redis.StrictRedis(host="localhost")):
        super(Replay, self).__init__()

        # main thread가 종료되면 그 즉시 종료되는 thread이다.
        self.setDaemon(True)

        self._memory = PrioritizedMemory(int(config.replayMemory))
        # self._memory = ReplayMemory(int(config.replayMemory))

        self._connect = connect
        self._connect.delete("sample")
        self._lock = threading.Lock()

    def run(self):
        t = 0
        while True:
            t += 1
            pipe = self._connect.pipeline()
            pipe.lrange("sample", 0, -1)
            pipe.ltrim("sample", -1, 0)
            data = pipe.execute()[0]
            if data is not None:
                for d in data:
                    dd, prior = loads(d)
                    with self._lock:
                        self._memory.push(dd, prior)
            time.sleep(0.01)
            if (t % 100) == 0:
                gc.collect()

    def sample(self, batch_size):
        with self._lock:
            return self._memory.sample(batch_size)

    def __len__(self):
        return len(self._memory)

    def update_priorities(self, indices, priorities):
        with self._lock:
            self._memory.update_priorities(indices, priorities)

    @property
    def total_prios(self):
        return self._memory.total_prios()

    def remove_to_fit(self):
        with self._lock:
            self._memory.remove_to_fit()