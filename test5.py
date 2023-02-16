import torch
import torch.nn as nn
input = torch.randn(3, 12, 4096)
lstm = nn.LSTM(4096, 512, 1, batch_first=True)
output, (h,c)  = lstm(input)

print(output.size())
print(h.size())
print(output[0][2]==h[0][0])
