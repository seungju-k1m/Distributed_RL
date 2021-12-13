from configuration import REDIS_SERVER, REDIS_SERVER_PUSH
import redis


c = redis.StrictRedis(host=REDIS_SERVER, port=6379)
k = redis.StrictRedis(host=REDIS_SERVER_PUSH, port=6379)

name = c.scan()

if len(name[-1]) > 0:
    c.delete(*name[-1])
    print(name[-1])
    print("DELETE")

name = k.scan()
if len(name[-1]) > 0:
    k.delete(*name[-1])
    print(name[-1])
    print("DELETE")