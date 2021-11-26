import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda"if torch.cuda.is_available()else"cpu")

EPSILON = 1e-5


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2, p_type):

    f_channel = channel_fusion(tensor1, tensor2, p_type)
    f_spatial = spatial_fusion(tensor1, tensor2, p_type)

    global_p_w1 = f_spatial / (f_spatial + f_channel + EPSILON)
    global_p_w2 = f_channel / (f_spatial + f_channel + EPSILON)

    tensor_f = global_p_w1 * f_spatial + global_p_w2 * f_channel

    return tensor_f


# select channel
def channel_fusion(tensor1, tensor2, p_type):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# channel attention
def channel_attention(tensor, pooling_type='avg'):
    # global pooling
    shape = tensor.size()

    c = shape[1]
    h = shape[2]
    w = shape[3]
    channel = torch.zeros(1, c, 1, 1)
    if pooling_type is"l1_mean":
        channel = torch.norm(tensor, p=1, dim=[2, 3], keepdim=True) / (h * w)
    elif pooling_type is"l2_mean":
        channel = torch.norm(tensor, p=2, dim=[2, 3], keepdim=True) / (h * w)
    elif pooling_type is "linf":
        ndarray = tensor.cpu().numpy()
        max = np.amax(ndarray,axis=(2,3))
        tensor = torch.from_numpy(max)
        channel = tensor.reshape(1,c,1,1)
        channel = channel.to(device)
    return channel


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = torch.zeros(1, 1, 1, 1)

    shape = tensor.size()
    c = shape[1]
    h = shape[2]
    w = shape[3]

    if spatial_type is 'l1_mean':
        spatial = torch.norm(tensor, p=1, dim=[1], keepdim=True) / c
    elif spatial_type is"l2_mean":
        spatial = torch.norm(tensor, p=2, dim=[1], keepdim=True) / c
    elif spatial_type is "linf":
        spatial, indices = tensor.max(dim=1, keepdim=True)
        spatial = spatial / c
        spatial = spatial.to(device)
    return spatial


