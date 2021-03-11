import redis

rd = redis.StrictRedis(host='localhost', port=6379, db=0)
rd.set(1, 100)