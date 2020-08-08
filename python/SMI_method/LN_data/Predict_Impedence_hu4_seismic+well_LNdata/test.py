import torch
import numpy as np


a = 14.0
a = np.int(a)
a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a)
b = a.view(1, -1, 10)
print(b)
c = b.view(1, 4, 5)
print(c)
