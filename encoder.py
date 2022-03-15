from json import encoder
from setup import *
def encoder_position(X, Dims):
    positions = [X]
    for i in range(Dims):
        for fn in [torch.sin, torch.cos]:
            positions.append(fn(2.0 ** i * X))
    return torch.concat(positions, axis= -1)