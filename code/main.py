from scipy import misc
import numpy as np
import torch
from einops import rearrange
import imageio
import pdb

import clustering


if __name__ == '__main__':

    img = torch.tensor(imageio.imread('checker-shadow/input-small.png'), dtype=torch.double).cuda()
    img = img / 255
    img = rearrange(img, 'h w c -> c h w')

    RED = [255, 0, 0]
    GREEN = [0, 255, 0]
    BLUE = [0, 0, 255]
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    
    CLASS_TO_COLOR = {
        0:  BLACK,
        1:  WHITE,
        2:  BLUE,
        3:  RED,
    }


    print('hello world')

    init_p = None

    for (p, mean, var, prior) in clustering.OverlapClustering().run_clustering(img, n_classes=2, radius=25, n_iter=10, stride=8, warmup_steps=2, warmup_radius=200, radius_steps=([200]*5), animate=True, color_map=None, init_p=init_p): init_p = p; print(p.shape)

    pdb.set_trace()


#    for x in OverlapClustering().run_clustering(img, n_classes=4, radius=25, n_iter=10, stride=8, warmup_steps=2, warmup_radius=200, radius_steps=([1000]*2 + [100]*13 + [50]*10)):
#        print(x[0].shape)

    

