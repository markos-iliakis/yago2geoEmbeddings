import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math

from module import *

# from module import get_activation_function, FCNet, coord_normalize

"""
A Set of position encoder
"""


def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # freq_list = []
        # for cur_freq in range(frequency_num):
        #     base = 1.0/(np.power(max_radius, cur_freq*1.0/(frequency_num-1)))
        #     freq_list.append(base)

        # freq_list = np.asarray(freq_list)

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
                                   (frequency_num * 1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = 1.0 / timescales

    return freq_list


"""
The theory based Grid cell spatial relation encoder, 
See https://openreview.net/forum?id=Syx0Mh05YQ
Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion
"""


class TheoryGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
                 max_radius=10000, min_radius=1000, freq_init="geometric",
                 ffn=None, device="cpu"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        # the frequency we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])  # 0
        self.unit_vec2 = np.asarray([-1.0 / 2.0, math.sqrt(3) / 2.0])  # 120 degree
        self.unit_vec3 = np.asarray([-1.0 / 2.0, -math.sqrt(3) / 2.0])  # 240 degree

        self.input_embed_dim = self.cal_input_dim()

        # self.f_act = get_activation_function(f_act, "TheoryGridCellSpatialRelationEncoder")
        # self.dropout = nn.Dropout(p=dropout)

        # self.use_post_mat = use_post_mat
        # if self.use_post_mat:
        #     self.post_linear_1 = nn.Linear(self.input_embed_dim, 64)
        #     nn.init.xavier_uniform(self.post_linear_1.weight)
        #     self.post_linear_2 = nn.Linear(64, self.spa_embed_dim)
        #     nn.init.xavier_uniform(self.post_linear_2.weight)
        #     self.dropout_ = nn.Dropout(p=dropout)
        # else:
        #     self.post_linear = nn.Linear(self.input_embed_dim, self.spa_embed_dim)
        #     nn.init.xavier_uniform(self.post_linear.weight)
        self.ffn = ffn

        self.device = device

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis=1)

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec 
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis=-1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis=-1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinusoid function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))

        # if self.use_post_mat:
        #     sprenc = self.post_linear_1(spr_embeds)
        #     sprenc = self.post_linear_2(self.dropout(sprenc))
        #     sprenc = self.f_act(self.dropout(sprenc))
        # else:
        #     sprenc = self.post_linear(spr_embeds)
        #     sprenc = self.f_act(self.dropout(sprenc))
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


