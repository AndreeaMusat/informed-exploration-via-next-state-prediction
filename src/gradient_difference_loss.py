import numpy as np
import torch
import torch.nn.functional as F


def gradient_difference_loss(gen_frames, gt_frames, alpha=1, reduction='mean'):
    def gradient(x):
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        dx, dy = right - left, bottom - top 
        
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    if reduction == 'mean':
        return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)    
    elif reduction == 'none':
        return grad_diff_x ** alpha + grad_diff_y ** alpha 

