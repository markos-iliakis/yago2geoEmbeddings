import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import torch.nn.functional as F

"""
A set of decoder modules.
Each decoder takes pairs of embeddings and predicts relationship scores given these embeddings.
"""

""" 
*Metapath decoders*
For all metapath encoders, the forward method returns a compositonal relationships score, 
i.e. the likelihood of compositonional relationship or metapath, between a pair of nodes.
"""


class BilinearBlockDiagMetapathDecoder(nn.Module):
    """
    This is only used for enc_agg_func == "concat"
    Each edge type is represented by two matrix:
    1) feature matrix for node feature embed
    2) position matrix for node position embed
    It can be seen as a block-diag matrix
    compositional relationships are a product matrices.
    """

    def __init__(self, relations, dims, feat_dims, spa_embed_dim):
        '''
        Args:
            relations: a dict() of all triple templates
                key:    domain entity type
                value:  a list of tuples (range entity type, predicate)
            dims: a dict(), node type => embed_dim of node embedding
            feat_dims: a dict(), node type => embed_dim of feature embedding
            spa_embed_dim: the embed_dim of position embedding
        '''
        super(BilinearBlockDiagMetapathDecoder, self).__init__()
        self.relations = relations
        self.dims = dims
        self.feat_dims = feat_dims
        self.spa_embed_dim = spa_embed_dim

        self.feat_mats = {}
        self.pos_mats = {}
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.feat_mats[rel] = nn.Parameter(torch.FloatTensor(feat_dims[rel[0]], feat_dims[rel[2]]))
                init.xavier_uniform_(self.feat_mats[rel])
                self.register_parameter("feat-" + "_".join(rel), self.feat_mats[rel])

                self.pos_mats[rel] = nn.Parameter(torch.FloatTensor(spa_embed_dim, spa_embed_dim))
                init.xavier_uniform_(self.pos_mats[rel])
                self.register_parameter("pos-" + "_".join(rel), self.pos_mats[rel])

    def forward(self, embeds1, embeds2, rels):
        '''
        embeds1, embeds2 shape: [embed_dim, batch_size]
        rels: a list of triple templates, a n-length metapath
        '''
        # act: [batch_size, embed_dim]
        act = embeds1.t()
        feat_act, pos_act = torch.split(act, [self.feat_dims[rels[0][0]], self.spa_embed_dim], dim=1)
        for i_rel in rels:
            feat_act = feat_act.mm(self.feat_mats[i_rel])
            pos_act = pos_act.mm(self.pos_mats[i_rel])
        #  act: [batch_size, embed_dim]
        act = torch.cat([feat_act, pos_act], dim=1)
        act = self.cos(act.t(), embeds2)
        return act

    def project(self, embeds, rel):
        '''
        embeds shape: [embed_dim, batch_size]
        rel: triple template
        '''
        feat_act, pos_act = torch.split(embeds.t(),
                                        [self.feat_dims[rel[0]], self.spa_embed_dim], dim=1)
        feat_act = feat_act.mm(self.feat_mats[rel])
        pos_act = pos_act.mm(self.pos_mats[rel])
        act = torch.cat([feat_act, pos_act], dim=1)
        return act.t()


"""
Set intersection operators.
"""


class SimpleSetIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors.
    Takes a simple element-wise min.
    """

    def __init__(self, agg_func=torch.min):
        super(SimpleSetIntersection, self).__init__()
        self.agg_func = agg_func

    # def forward(self, embeds1, embeds2, mode, embeds3 = []):
    #     if len(embeds3) > 0:
    #         combined = torch.stack([embeds1, embeds2, embeds3])
    #     else:
    #         combined = torch.stack([embeds1, embeds2])
    #     aggs = self.agg_func(combined, dim=0)
    #     if type(aggs) == tuple:
    #         aggs = aggs[0]
    #     return aggs, combined

    def forward(self, mode, embeds_list):
        if len(embeds_list) < 2:
            raise Exception("The intersection needs more than one embeding")

        combined = torch.stack(embeds_list)
        aggs = self.agg_func(combined, dim=0)
        if type(aggs) == tuple:
            aggs = aggs[0]
        return aggs, combined
