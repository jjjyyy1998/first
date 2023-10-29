# import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

class SmoothGrad():
    def __init__(self, model, use_cuda, stdev_spread=0.15, n_samples=100, magnitude=True):
        self.model = model.eval()
        self.device = use_cuda
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        self.model = self.model.to(self.device)

    def __call__(self, x,y, index=None):
        for i in range(len(x)):
            target = x[i].clone()
            target = target.to(self.device)
            total_gradients = torch.zeros_like(target)
            true = y 
            true = int(true)
            # print(k.shape)
            target = target.view(-1,target.size()[0],target.size()[1],target.size()[2])
            # print(k.shape)
            stdev = self.stdev_spread/(target.max() - target.min())
            std_tensor = torch.ones_like(target) * stdev
            x_plus_noise = torch.normal(mean=target, std=std_tensor)
            x_plus_noise.requires_grad_()
            x_plus_noise = x_plus_noise.to(self.device)
            print(x_plus_noise.shape)
            l,output,__,_,f = self.model(x_plus_noise)
            one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
            _,preds = torch.max(output,1)
            if true != preds or true == preds:

                for j in tqdm(range(self.n_samples)):
                    x_plus_noise = torch.normal(mean=target, std=std_tensor)
                    x_plus_noise.requires_grad_()
                    x_plus_noise = x_plus_noise.to(self.device)
                    l,output,__,emb,f = self.model(x_plus_noise)
                    one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
                    _,preds = torch.max(output,1)
                    one_hot[0][true] = 1
                    one_hot = torch.from_numpy(one_hot)
                    # one_hot = (true - output)**2
                    one_hot = torch.sum(one_hot.to(self.device) * output)
        
                    one_hot = torch.sum(emb)
                    
                    # one_hot = torch.mean(one_hot)
                    # print(one_hot)
                    # one_hot.to(self.device)
                    # print(one_hot)
                    # else:
                    #     # one_hot = torch.sum(one_hot * output)
                    
                    if x_plus_noise.grad is not None:
                        x_plus_noise.grad.data.zero_()
                        # print('error')
                    one_hot.backward()
                    
                    grad = x_plus_noise.grad
                    # print(grad.shape)
                    
                    if self.magnitude:
                        total_gradients += (grad * grad)[0]
                    else:
                        total_gradients += grad
                    #if torch.sum(grad) == 0:
                        # print(1)
                        
                avg_gradients = total_gradients/ self.n_samples
                # for i in range(avg_gradients.shape[0]):
                #     if torch.sum(avg_gradients[i]) == 0:
                #         print(i)
                avg_gradients = avg_gradients.cpu().numpy()
                # for i in range(avg_gradients.shape[0]):
                #     if np.sum(avg_gradients[i]) == 0:
                #         print(i)
                # print(avg_gradients)
                break
            else:
                continue
        return avg_gradients,i



def show_as_gray_image(img,percentile=99):
    img_2d = np.sum(img, axis=2)
    span = abs(np.percentile(img_2d, percentile))
    vmin = -span
    vmax = span
    img_2d = np.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
    print(img_2d)
    return np.uint8(img_2d*255)
