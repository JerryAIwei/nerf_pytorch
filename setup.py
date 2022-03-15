import profile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda")
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile = "full")
