from baseline.utils import jsonParser, writeTrainInfo
from baseline.utils import setup_logger
from datetime import datetime, timedelta

import logging
import os
import torch
import math


# _path_ = "./cfg/ape_x.json"
_path_ = "./cfg/impala.json"
# _path_ = './cfg/r2d2.json'


_parser_ = jsonParser(_path_)
_data_ = _parser_.loadParser()

ALG = _data_['ALG']


DATA = _data_

# APE_X

if ALG == "APE_X":
    USE_REWARD_CLIP = _data_["USE_REWARD_CLIP"]
    BASE_PATH = "./log/APE_X"

# R2D2
elif ALG == "R2D2":
    FIXED_TRAJECTORY = _data_["FIXED_TRAJECTORY"]
    MEM = _data_["MEM"]
    USE_RESCALING = _data_["USE_RESCALING"]
    BASE_PATH = "./log/R2D2"

elif ALG == "IMPALA":
    C_LAMBDA = _data_["C_LAMBDA"]
    C_VALUE = _data_["C_VALUE"]
    P_VALUE = _data_["P_VALUE"]
    ENTROPY_R = _data_["ENTROPY_R"]
    BASE_PATH = ',/log/IMPALA'

# COMMOLN

use_per = ALG != "IMPALA"

if use_per:

    ALPHA = _data_['ALPHA']
    BETA = _data_['BETA']
    TARGET_FREQUENCY = _data_['TARGET_FREQUENCY']
    N = _data_['N']

GAMMA = _data_['GAMMA']
BATCHSIZE = _data_['BATCHSIZE']
ACTION_SIZE = _data_['ACTION_SIZE']
UNROLL_STEP = _data_['UNROLL_STEP']
REPLAY_MEMORY_LEN = _data_["REPLAY_MEMORY_LEN"]

REDIS_SERVER = _data_["REDIS_SERVER"]
try:
    REDIS_SERVER_PUSH = _data_["REDIS_SERVER_PUSH"]
except:
    pass

DEVICE = _data_["DEVICE"]
LEARNER_DEVICE = _data_["LEARNER_DEVICE"]

BUFFER_SIZE = _data_["BUFFER_SIZE"]


# MODEL & OPTIM

OPTIM_INFO = _data_["optim"]
MODEL = _data_["model"]


# SAVE and LOG PATH
_current_time_ = datetime.now()
CURRENT_TIME = _current_time_.strftime("%m_%d_%Y_%H_%M_%S")