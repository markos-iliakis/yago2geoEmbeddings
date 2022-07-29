import random
from SpatialRelationEncoder import *

"""
Set of modules for encoding nodes.
These modules take as input node ids and output embeddings.
"""


class DirectEncoder(nn.Module):
    """
    Encodes a node as a embedding via direct lookup.
    (i.e., this is just like basic node2vec or matrix factorization)
    """

    def __init__(self, features, feature_modules):
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_modules  -- This should be a map from mode -> torch.nn.EmbeddingBag 

        features(nodes, mode): a embedding lookup function to make a dict() from node type to embeddingbag
            nodes: a lists of global node id which are in type (mode)
            mode: node type
            return: embedding vectors, shape [num_node, embed_dim]
        feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
        """
        super(DirectEncoder, self).__init__()
        for name, module in feature_modules.items():
            self.add_module("feat-" + name, module)
        self.features = features

    def forward(self, nodes, mode, offset=None, **kwargs):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        mode      -- string designating the mode of the nodes
        offsets   -- specifies how the embeddings are aggregated. 
                     see torch.nn.EmbeddingBag for format. 
                     No aggregation if offsets is None
        """

        if offset is None:
            # the output is a dict() map: node type --> embedding tensor [num_ent, embed_dim]
            # t() transpose embedding tensor as [embed_dim, num_ent]
            embeds = self.features(nodes, mode).t()
            # calculate the L2-norm for each embedding vector, [1, num_ent]
            norm = embeds.norm(p=2, dim=0, keepdim=True)
            # normalize the embedding vectors
            # shape: [embed_dim, num_ent]
            # return embeds.div(norm.expand_as(embeds))
            return embeds.div(norm)
        else:
            return self.features(nodes, mode, offset).t()


def geo_lookup(nodes, id2geo, add_dim=-1, id2extent=None, doExtentSample=False):
    '''
    Given a list of node id, make a coordinate tensor and a nogeo indicator tensor
    Args:
        nodes: list of nodes id
        id2geo: a dict()
            key: node id
            value: a list, [longitude, lantitude]
    Return:
        coord_tensor: [batch_size, 2], geographic coordicate for geo_ent, [0.0, 0.0] for nogeo_ent
        nogeo_khot: [batch_size], 0 for geo_ent, 1 for nogeo_ent
    '''
    coord_tensor = []
    nogeo_khot = []
    for i, eid in enumerate(nodes):
        if eid in id2geo:
            if id2extent is None:
                coords = list(id2geo[eid])
            else:
                if eid in id2extent:
                    if doExtentSample:
                        xmin, xmax, ymin, ymax = id2extent[eid]
                        x = random.uniform(xmin, xmax)
                        y = random.uniform(ymin, ymax)
                        coords = [x, y]
                    else:
                        coords = list(id2geo[eid])
                else:
                    coords = list(id2geo[eid])
            if add_dim == -1:
                coord_tensor.append(coords)
            elif add_dim == 1:
                coord_tensor.append([coords])
            nogeo_khot.append(0)
        else:
            if add_dim == -1:
                coord_tensor.append([0.0, 0.0])
            elif add_dim == 1:
                coord_tensor.append([[0.0, 0.0]])
            nogeo_khot.append(1)

    return coord_tensor, nogeo_khot


class ExtentPositionEncoder(nn.Module):
    '''
    This is position encoder, a wrapper for different space encoder,
    Given a list of node ids, return their embedding
    '''

    def __init__(self, spa_enc_type, id2geo, id2extent, spa_enc, graph, spa_enc_embed_norm, device="cpu"):
        '''
        Args:
            out_dims: a dict()
                key: node type
                value: embedding dimention
            
            spa_enc_type: the type of space encoder
            id2geo: a dict(): node id -> [longitude, latitude]
            id2extent: a dict(): node id -> (xmin, xmax, ymin, ymax)
            spa_enc: one space encoder
            graph: Graph()
            spa_enc_embed_norm: whether to do position embedding normalization
            
            
        '''
        super(ExtentPositionEncoder, self).__init__()
        self.spa_enc_type = spa_enc_type
        self.id2geo = id2geo
        self.id2extent = id2extent
        self.spa_embed_dim = spa_enc.spa_embed_dim  # the output space embedding
        self.spa_enc = spa_enc
        self.graph = graph
        self.spa_enc_embed_norm = spa_enc_embed_norm
        self.device = device

        self.nogeo_idmap = self.make_nogeo_idmap(self.id2geo, self.graph)
        # random initialize the position embedding for nogeo entities
        # last index: indicate the geo-entity, use this for convinience
        self.nogeo_spa_embed_module = torch.nn.Embedding(len(self.nogeo_idmap) + 1, self.spa_embed_dim).to(self.device)
        self.add_module("nogeo_pos_embed_matrix", self.nogeo_spa_embed_module)
        # define embedding initialization method: normal dist
        self.nogeo_spa_embed_module.weight.data.normal_(0, 1. / self.spa_embed_dim)

    def nogeo_embed_lookup(self, nodes):
        '''
        nogeo_spa_embeds: the spa embed for no-geo entity, [batch_size, spa_embed_dim]
        Note for geo-entity, we use the last embedding in self.nogeo_spa_embed_module
        
        '''
        id_list = []
        for node in nodes:
            if node in self.nogeo_idmap:
                # if this is nogeo entity
                id_list.append(self.nogeo_idmap[node])
            else:
                # if this is geo entity
                id_list.append(len(self.nogeo_idmap))

        # nogeo_spa_embeds: (batch_size, spa_embed_dim)
        nogeo_spa_embeds = self.nogeo_spa_embed_module(
            torch.autograd.Variable(torch.LongTensor(id_list).to(self.device)))
        # calculate the L2-norm for each embedding vector, (batch_size, 1)
        norm = nogeo_spa_embeds.norm(p=2, dim=1, keepdim=True)
        # normalize the embedding vectors
        # shape: [batch_size, spa_embed_dim]
        return nogeo_spa_embeds.div(norm)

    def make_nogeo_idmap(self, id2geo, graph):
        '''
        nogeo_idmap: dict(), nogeo-entity id => local id
        '''
        id_set = set()
        for mode in graph.full_sets:
            id_set.union(graph.full_sets[mode])
        # id_list = sorted(list(id_set))
        geo_set = set(id2geo.keys())
        nogeo_set = id_set - geo_set
        nogeo_idmap = {nogeo_id: i for i, nogeo_id in enumerate(nogeo_set)}
        return nogeo_idmap

    def forward(self, nodes, do_test=False):
        '''
        Args:
            nodes: a list of node ids
        Return:
            pos_embeds: the position embedding for all nodes, (spa_embed_dim, batch_size)
                    geo_ent => space embedding from geographic coordinates
                    nogeo_ent => [0,0..,0]
        '''
        # coord_tensor: [batch_size, 1, 2], geographic coordicate for geo_ent, [0.0, 0.0] for nogeo_ent
        # nogeo_khot: [batch_size], 0 for geo_ent, 1 for nogeo_ent
        coord_tensor, nogeo_khot = geo_lookup(nodes,
                                              id2geo=self.id2geo,
                                              add_dim=1,
                                              id2extent=self.id2extent,
                                              doExtentSample=True)

        # spa_embeds: (batch_size, 1, spa_embed_dim)
        spa_embeds = self.spa_enc(coord_tensor)
        # spa_embeds: (batch_size, spa_embed_dim)
        spa_embeds = torch.squeeze(spa_embeds, dim=1)

        nogeo_khot = torch.FloatTensor(nogeo_khot).to(self.device)
        # mask: (batch_size, 1)
        mask = torch.unsqueeze(nogeo_khot, dim=1)
        # pos_embeds: (batch_size, spa_embed_dim), erase nogeo embed as [0,0...]
        pos_embeds = spa_embeds * (1 - mask)

        # nogeo_spa_embeds: (batch_size, spa_embed_dim)
        nogeo_spa_embeds = self.nogeo_embed_lookup(nodes)
        # nogeo_pos_embeds: (batch_size, spa_embed_dim), erase geo embed as [0,0...]
        nogeo_pos_embeds = nogeo_spa_embeds * mask

        # pos_embeds: (batch_size, spa_embed_dim)
        pos_embeds = pos_embeds + nogeo_pos_embeds

        # pos_embeds: (spa_embed_dim, batch_size)
        pos_embeds = pos_embeds.t()

        if self.spa_enc_embed_norm:
            # calculate the L2-norm for each embedding vector, (1, batch_size)
            norm = pos_embeds.norm(p=2, dim=0, keepdim=True)
            # normalize the embedding vectors
            # shape: (spa_embed_dim, batch_size)
            return pos_embeds.div(norm)

        return pos_embeds


class PositionEncoder(nn.Module):
    '''
    This is position encoder, a wrapper for different space encoder,
    Given a list of node ids, return their embedding
    '''

    def __init__(self, spa_enc_type, id2geo, spa_enc, graph, spa_enc_embed_norm, device="cpu"):
        '''
        Args:
            out_dims: a dict()
                key: node type
                value: embedding dimention
            
            spa_enc_type: the type of space encoder
            id2geo: a dict(): node id -> [longitude, latitude]
            spa_enc: one space encoder
            graph: Graph()
            spa_enc_embed_norm: whether to do position embedding normalization
            
            
        '''
        super(PositionEncoder, self).__init__()
        self.spa_enc_type = spa_enc_type
        self.id2geo = id2geo
        self.spa_embed_dim = spa_enc.spa_embed_dim  # the output space embedding
        self.spa_enc = spa_enc
        self.graph = graph
        self.spa_enc_embed_norm = spa_enc_embed_norm
        self.device = device

        self.nogeo_idmap = self.make_nogeo_idmap(self.id2geo, self.graph)
        # random initialize the position embedding for nogeo entities
        # last index: indicate the geo-entity, use this for convinience
        self.nogeo_spa_embed_module = torch.nn.Embedding(len(self.nogeo_idmap) + 1, self.spa_embed_dim).to(self.device)
        self.add_module("nogeo_pos_embed_matrix", self.nogeo_spa_embed_module)
        # define embedding initialization method: normal dist
        self.nogeo_spa_embed_module.weight.data.normal_(0, 1. / self.spa_embed_dim)

    def nogeo_embed_lookup(self, nodes):
        '''
        nogeo_spa_embeds: the spa embed for no-geo entity, [batch_size, spa_embed_dim]
        Note for geo-entity, we use the last embedding in self.nogeo_spa_embed_module
        
        '''
        id_list = []
        for node in nodes:
            if node in self.nogeo_idmap:
                # if this is nogeo entity
                id_list.append(self.nogeo_idmap[node])
            else:
                # if this is geo entity
                id_list.append(len(self.nogeo_idmap))

        # nogeo_spa_embeds: (batch_size, spa_embed_dim)
        nogeo_spa_embeds = self.nogeo_spa_embed_module(
            torch.autograd.Variable(torch.LongTensor(id_list).to(self.device)))
        # calculate the L2-norm for each embedding vector, (batch_size, 1)
        norm = nogeo_spa_embeds.norm(p=2, dim=1, keepdim=True)
        # normalize the embedding vectors
        # shape: [batch_size, spa_embed_dim]
        return nogeo_spa_embeds.div(norm.expand_as(nogeo_spa_embeds))

    def make_nogeo_idmap(self, id2geo, graph):
        '''
        nogeo_idmap: dict(), nogeo-entity id => local id
        '''
        id_set = set()
        for mode in graph.full_sets:
            id_set.union(graph.full_sets[mode])
        # id_list = sorted(list(id_set))
        geo_set = set(id2geo.keys())
        nogeo_set = id_set - geo_set
        nogeo_idmap = {nogeo_id: i for i, nogeo_id in enumerate(nogeo_set)}
        return nogeo_idmap

    def forward(self, nodes):
        '''
        Args:
            nodes: a list of node ids
        Return:
            pos_embeds: the position embedding for all nodes, (spa_embed_dim, batch_size)
                    geo_ent => space embedding from geographic coordinates
                    nogeo_ent => [0,0..,0]
        '''
        # coord_tensor: [batch_size, 1, 2], geographic coordicate for geo_ent, [0.0, 0.0] for nogeo_ent
        # nogeo_khot: [batch_size], 0 for geo_ent, 1 for nogeo_ent
        coord_tensor, nogeo_khot = geo_lookup(nodes, self.id2geo, add_dim=1)

        # spa_embeds: (batch_size, 1, spa_embed_dim)
        spa_embeds = self.spa_enc(coord_tensor)
        # spa_embeds: (batch_size, spa_embed_dim)
        spa_embeds = torch.squeeze(spa_embeds, dim=1)

        nogeo_khot = torch.FloatTensor(nogeo_khot).to(self.device)
        # mask: (batch_size, 1)
        mask = torch.unsqueeze(nogeo_khot, dim=1)
        # pos_embeds: (batch_size, spa_embed_dim), erase nogeo embed as [0,0...]
        pos_embeds = spa_embeds * (1 - mask)

        # nogeo_spa_embeds: (batch_size, spa_embed_dim)
        nogeo_spa_embeds = self.nogeo_embed_lookup(nodes)
        # nogeo_pos_embeds: (batch_size, spa_embed_dim), erase geo embed as [0,0...]
        nogeo_pos_embeds = nogeo_spa_embeds * mask

        # pos_embeds: (batch_size, spa_embed_dim)
        pos_embeds = pos_embeds + nogeo_pos_embeds

        # pos_embeds: (spa_embed_dim, batch_size)
        pos_embeds = pos_embeds.t()

        if self.spa_enc_embed_norm:
            # calculate the L2-norm for each embedding vector, (1, batch_size)
            norm = pos_embeds.norm(p=2, dim=0, keepdim=True)
            # normalize the embedding vectors
            # shape: (spa_embed_dim, batch_size)
            return pos_embeds.div(norm)

        return pos_embeds


'''
End for space encoding
'''


########################


class NodeEncoder(nn.Module):
    """
    This is the encoder for each entity or node which has two components"
    1. feature encoder (DirectEncoder): feat_enc
    2. position encoder (PositionEncoder): pos_enc
    """

    def __init__(self, feat_enc, pos_enc, agg_type="add"):
        '''
        Args:
            feat_enc:feature encoder
            pos_enc: position encoder
            agg_type: how to combine the feature embedding and space embedding of a node/entity
        '''
        super(NodeEncoder, self).__init__()
        self.feat_enc = feat_enc
        self.pos_enc = pos_enc
        self.agg_type = agg_type
        if feat_enc is None and pos_enc is None:
            raise Exception("pos_enc and feat_enc are both None!!")

    def forward(self, nodes, mode, offset=None):
        '''
        Args:
            nodes: a list of node ids
        Return:
            
            embeds: node embedding
                if agg_type in ["add", "min", "max", "mean"]:
                    # here we assume spa_embed_dim == embed_dim 
                    shape [embed_dim, num_ent]
                if agg_type == "concat":
                    shape [embed_dim + spa_embed_dim, num_ent]
        '''
        if self.feat_enc is not None and self.pos_enc is not None:
            # we have both feature encoder and position encoder

            # feat_embeds: [embed_dim, num_ent]
            feat_embeds = self.feat_enc(nodes, mode, offset=offset)

            # pos_embeds: [embed_dim, num_ent]
            pos_embeds = self.pos_enc(nodes)
            if self.agg_type == "add":
                embeds = feat_embeds + pos_embeds
            elif self.agg_type in ["min", "max", "mean"]:

                if self.agg_type == "min":
                    agg_func = torch.min
                elif self.agg_type == "max":
                    agg_func = torch.max
                elif self.agg_type == "mean":
                    agg_func = torch.mean
                combined = torch.stack([feat_embeds, pos_embeds])
                aggs = agg_func(combined, dim=0)
                if type(aggs) == tuple:
                    aggs = aggs[0]
                embeds = aggs
            elif self.agg_type == "concat":
                embeds = torch.cat([feat_embeds, pos_embeds], dim=0)
            else:
                raise Exception("The Node Encoder Aggregation type is not supported!!")
        elif self.feat_enc is None and self.pos_enc is not None:
            # we only have position encoder

            # pos_embeds: [embed_dim, num_ent]
            pos_embeds = self.pos_enc(nodes)

            embeds = pos_embeds
        elif self.feat_enc is not None and self.pos_enc is None:
            # we only have feature encoder

            # feat_embeds: [embed_dim, num_ent]
            feat_embeds = self.feat_enc(nodes, mode, offset=offset)

            embeds = feat_embeds

        return embeds