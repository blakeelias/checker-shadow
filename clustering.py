import time
import os
import pdb

import torch
import numpy as np
from einops import rearrange

import imageio
from matplotlib import pyplot as plt
from PIL import Image

from training.pytorch.utils import save_visualize


class OverlapClustering():

    def __init__(self, gpuid=0):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

        self.device = torch.device('cuda:0')
        

    def local_avg(self, data, radius, stride=1):
            w,h = data.shape[-2:]
            mean = torch.nn.functional.avg_pool2d(data.reshape(-1,1,w,h), 2*radius+1, stride=stride, padding=radius, count_include_pad=False)
            if stride>1: mean = torch.nn.functional.interpolate(mean, size=(w,h), mode='bilinear')
            return mean.view(data.shape)

    def local_moments(self, data, q, radius, stride=1, var_min=0.0001, mq_min=0.000001):
        mq = self.local_avg(q, radius, stride)
        mq.clamp(min=mq_min)
        weighted = torch.einsum('zij,cij->czij', data, q) #class,channel,x,y
        weighted_sq = torch.einsum('zij,cij->czij', data**2, q)
        mean = self.local_avg(weighted, radius, stride) / mq.unsqueeze(1)
        var = self.local_avg(weighted_sq, radius, stride) / mq.unsqueeze(1) - mean**2
        var = var.clamp(min=var_min)
        return mean, var

    def lp_gaussian(self, data, mean, var, radius, stride=1):
        #means: c,ch,x,y
        #data: ch,x,y
        #out: c,x,y
        m0 = -(1 / var)
        m1 = (2 * mean / var)
        m2 = -(mean**2 / var)
        L = (torch.log(var)) / 2 # fix to compute log(sigma), not log(sigma^2)

        return (m0*data**2 + m1*data + m2 - 2 * L).sum(1) / 2
    
    def prob_gaussian(self, data, prior, mean, var, radius, stride=1):
        lp = self.lp_gaussian(data, mean, var, radius, stride)
        p = lp.softmax(0) * prior
        p /= p.sum(0)
        p+=0.001
        p /= p.sum(0)
        return p

    def em(self, data, p, radius, stride=1):
        prior = self.local_avg(p, radius, stride)
        mean, var = self.local_moments(data, p, radius, stride)
        p_new = self.prob_gaussian(data, prior, mean, var, radius, stride)
        return p_new, mean, var, prior

    def run_clustering(self, image, n_classes, radius, n_iter, stride, warmup_steps, warmup_radius, radius_steps, animate=False, color_map=None, init_p=None):
        t = time.time()
        
        data = torch.tensor(image).to(self.device)
        if init_p is not None:
            p = torch.tensor(init_p).to(self.device)
        else:
            p = torch.rand((n_classes,) + image.shape[1:], dtype=torch.double).to(self.device)
            p /= p.sum(0)

        n_iter = len(radius_steps)

        # Initialize animation
        outputs = []
        
        
        for i in range(n_iter):
            p, mean, var, prior = self.em(data, p, radius_steps[i], stride)# stride if i<n_iter-1 else 1)

            p_ = p.cpu().numpy()
            mean_ = mean.cpu().numpy()
            var_ = var.cpu().numpy()
            prior_ = prior.cpu().numpy()
            
            # output_clusters_hard = p_.argmax(axis=0)
            # output_clusters_img = save_visualize.classes_to_rgb(output_clusters_hard, color_map)
            if animate:
                new_output = save_visualize.classes_to_rgb(rearrange(p_.argmax(axis=0), 'h w -> () h w'), color_map)[0]
                
                plt.imshow(rearrange(new_output.numpy(), 'c h w -> h w c'))
                plt.pause(0.05)
            else:
                outputs.append(p_)
                
            yield p_, mean_, var_, prior_



        if not animate:
            base_path = '/mnt/blobfuse/pred-output/overlap-clustering/shadow'
            previous_saves = [int(subdir) for subdir in os.listdir(base_path) if represents_int(subdir)]
            if len(previous_saves) > 0:
                last_save = max(previous_saves)
            else:
                last_save = 0
            path = '%s/%d' % (base_path, last_save + 1)

            inputs = torch.tensor([image])
            outputs = torch.tensor(outputs)
        
            # save_visualize.save_batch(outputs, path, 'output_clustering')
            # save_visualize.save_batch(inputs_visualize, path, 'input_clustering')
            save_visualize.save_visualize(inputs, outputs, None, path, rand_colors=True)
        
        p_ = p.cpu().numpy()
        mean_ = mean.cpu().numpy()
        var_ = var.cpu().numpy()
        prior_ = prior.cpu().numpy()
        
        print('total time: ', time.time() - t)
        
        return p_, mean_, var_, prior_
    


if __name__ == '__main__':
    img = imageio.imread('checker-shadow/input-small.png')
    img = img / 255
    img = rearrange(img, 'h w c -> c h w')
    
    print('hello world')
    
    for x in OverlapClustering().run_clustering(img, n_classes=3, radius=25, n_iter=10, stride=8, warmup_steps=2, warmup_radius=200, radius_steps=([1000]*2 + [20]*13)):
        print(x[0].shape)
