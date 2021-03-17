import torch
from baseline.baseAgent import baseAgent

netData = {
    "module00": {
        "netCat": "LSTM",
        "iSize": 256,
        "nLayer": 1,
        "hiddenSize": 64,
        "prior": 0,
        "input": [0, 1],
    },
    "module01": {
        "netCat": "Select",
        "num": 0,
        "prior": 1,
        "prevNodeNames": ["module00"],
        "output": True,
    },
    "module02": {
        "netCat": "Select",
        "num": 1,
        "prior": 2,
        "prevNodeNames": ["module00"],
        "output": True,
    },
}

print(
    """

This is test for checking the valid forward of LSTM and select module

input: 1, 1, 256 dim tensor
hidden state : 1, 64 dim tensor

"""
)
_input = torch.rand((1, 1, 256)).float()
(hx, nx) = torch.rand((1, 1, 64)).float(), torch.rand((1, 1, 64)).float()


def test_LSTM():
    agent = baseAgent(netData)
    shc = (_input, (hx, nx))
    s = (_input)
    _output = agent.forward(shc)
    print(_output[0], _output[1])
