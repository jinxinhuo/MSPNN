import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.4):
    n_class = 2

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

# 0.06  NT  74.64  Faceswap 91.44
# 0.1 NT 74.26   Faceswap 91.83
# 0.15 NT 74.61  Faceswap 91.72
# 0.15 NT 加上 b 74.87
# 0.2 NT 74.79 (64epoch)
# 0.3 NT 74.99 (65epoch)
# 0.4 NT 75 (65\90epoch)
