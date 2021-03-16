import redis

_connect = redis.StrictRedis(host="localhost")
_connect.delete("sample")
_connect.delete("Reward")
_connect.delete("params")
_connect.delete("Count")