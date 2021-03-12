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

        self._connect = connect
        self._connect.delete("sample")
        self._lock = threading.Lock()

    def run(self):
        while True:
            pipe = self._connect.pipeline()
            pipe.lrange("sample", 0, -1)
            pipe.ltrim("sample", -1, 0)
            data = pipe.execute()[0]
            if data is not None:
                for d in data:
                    p, t = loads(d)
                    self._memory.push(t, p)
            time.sleep(0.01)

    def update_priorities(self, indices, priorities):
        with self._lock:
            self._memory.update_priorities(indices, priorities)

    def remove_to_fit(self):
        with self._lock:
            self._memory.remove_to_fit()

    def sample(self, batch_size):
        with self._lock:
            return self._memory.sample(batch_size)

    def __len__(self):
        return len(self._memory)

    @property
    def total_prios(self):
        return self._memory.total_prios()
