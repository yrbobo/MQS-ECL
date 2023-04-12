import torch
import torch.nn as nn


class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()
        return

    def forward(self, features, mask):
        batch_size, token_length, embed_size = features.shape
        feature_mask = mask.view(batch_size, -1, 1).repeat(1, 1, embed_size)
        feature_mask_count = torch.sum(feature_mask, dim=1)
        return torch.sum(features * feature_mask, dim=1) / feature_mask_count
