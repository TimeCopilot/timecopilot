import os

import torch

os.environ["CUDA_HOME"] = os.path.join(torch.__path__[0], "lib")
