# Distributed Reinforcment Learning.

## Description

![](./docs/show.png)
This Repo is for implementation of Distributed RL Algorithm,

using by **Pytorch, Ray and Redis.**

Here is list of Algorithm I implemented (or will implement)

## Algorithms

1. IMPALA

2. APE_X_DQN

3. R2D2

---------------------------
## Google Cloud Platform

In APE-X DQN paper, the computer resources for experiment is below:

    1. nCpus: 360+
    2. RAM:256GB
    3. GPU: V100
    4. High Performance CPU for constructing data pipeline

Probably, Most pepole can't satsify above conditions.

Instead of buying it, I use Virtual Machine in GCP.

GCP supports not only for enough resoruces but also for redis-server.

-------------------------------


## Install

    Recommend you create the new development conda env for this repo.

    conda create -n <env_name> python=3.7

    git clone https://github.com/seungju-mmc/Distributed_RL.git

    git submodule init

    #  pull submodule from git 'baseline'
    #  If you read Readme.md from baseline, understand what it is.

    git submodule update
    
    pip install -r requirements.txt


**[Important] you must check ./cfg/<algorithm>.json. you can control the code by .json.**

**[Important] In configuration.py,  set the path !!**


 ## Run

    #  you need independent two terminals. Each one is for learner and actor.

    sudo apt-get install tmux

    tmux

    python run_learner.py

    # Crtl + b, then d

    tmux

    python run_actor.py --num-worker <n>


## BottleNecks

    1. More Cache memory, Better performance

        I observed that lower cache memory of intel i7 9700k (12Mb) can be siginificant bottleneck for constructing data pipeline. 
        


        You can check following lines in bash

    
    sudo lshw -C memory