# To check if the system can support the current downloaded torch cuda version.

import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))

x = torch.rand(1000, 1000, device="cuda")
print((x @ x).sum().item())