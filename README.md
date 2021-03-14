# Distributed Reinforcment Learning.

## Description

![](./docs/img/distributedRL_Str.jpeg)

This Repo is for implementation of Distributed RL,

using by **Pytorch, Ray and Redis.**

Here is list of Algorithm I implemented (or will implement)

## Algorithms

1. Soft Actor Critic

2. Proximal Policy Optimization

3. Muzero

4. R2D2

## Install

    Recommend you create the new development conda env for this repo.

    conda create -n <env_name> python=3.7

    git submodule init

    #  pull submodule from git 'baseline'
    #  If you read Readme.md from baseline, understand what it is.
    git submodule update
    

    pip install -r requirements.txt


then, if you use mujoco environment, you must install mujoco-py from this.