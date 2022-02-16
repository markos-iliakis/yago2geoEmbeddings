import pickle

import os
import torch

import graph
from Data.data_creator import read_graph, read_id2geo
from SpatialRelationEncoder import TheoryGridCellSpatialRelationEncoder
from attention import IntersectConcatAttention
from decoders import BilinearBlockDiagMetapathDecoder, SimpleSetIntersection
from encoders import DirectEncoder, ExtentPositionEncoder, NodeEncoder
from model import QueryEncoderDecoder
from module import MultiLayerFeedForwardNN
from trainer import make_args_parser, Trainer, make_args_combine
from utils import eval_perc_queries


def my_test(trainer):
    # Which district is near Hitchin ? -> Cadwell
    # 4005|('<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>', 4, '<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>')|3995
    # <http://kr_di_uoa_gr/yago2geo/resource/osentity_Cadwell_Ward_4005> 4005
    # <http://www_opengis_net/ont/geosparql#sfTouches> 4
    # <http://kr_di_uoa_gr/yaGeoYAGOgo2geo/resource/osentity_Hitchin_Oughton_Ward_3995> 3995

    # Which district is Catford inside of ?
    # 11063|('<http://kr_di_uoa_gr/yago2geo/ontology/OS_LondonBoroughWard>', 6, '<http://kr_di_uoa_gr/yago2geo/ontology/OS_EuropeanRegion>')|41428
    # <http://kr_di_uoa_gr/yago2geo/resource/osentity_Catford_South_Ward_11036> 11036
    # <http://www_opengis_net/ont/geosparql#sfWithin> 6
    # <http://kr_di_uoa_gr/yago2geo/resource/osentity_London_Euro_Region_41428> 41428

    # 30022|('<http://kr_di_uoa_gr/yago2geo/ontology/OS_UnitaryAuthority>', 3, '<http://kr_di_uoa_gr/yago2geo/ontology/OS_EuropeanRegion>')|41429
    # <http://kr_di_uoa_gr/yago2geo/resource/osentity_Sgir'Uige_agus_Ceann_a_Tuath_nan_Loch_Ward_43381> 43381
    # <http://www_opengis_net/ont/geosparql#sfIntersects> 3
    # <http://kr_di_uoa_gr/yago2geo/resource/osentity_Scotland_Euro_Region_41429> 41429

    f = graph.Formula('1-chain', (('<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>', '4', '<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>'),))
    query = graph.Query(('1-chain', ('4005', ('<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>', '4', '<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>'), '3995')))


if __name__ == '__main__':
    raw_info = pickle.load(open('./Data/test_edges.pkl', 'rb'))
    for raw_query in raw_info:
        if raw_query[0][1][0] == '4005' and raw_query[0][1][2] == '3995':
            # negs = raw_query[1]
            query = graph.Query.deserialize(raw_query, keep_graph=False)
            break

    formula = graph.Formula('1-chain', (('<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>', '4', '<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>'),))

    print("Loading graph data..")
    parser = make_args_parser()
    args = parser.parse_args()

    graph_path ='./Data/graph.pkl'

    graph, feature_modules, node_maps = read_graph(graph_path)
    out_dims = {e_type: 64 for e_type in graph.relations}
    model_out_dims = {mode: out_dims[mode] + 64 for mode in out_dims}

    print('Creating Encoder Operator..')
    # encoder
    feat_enc = DirectEncoder(graph.features, feature_modules)
    ffn = MultiLayerFeedForwardNN(input_dim=6*16, output_dim=64, num_hidden_layers=1, dropout_rate=0.5, hidden_dim=512, activation='sigmoid', use_layernormalize=True, skip_connection=True, context_str='TheoryGridCellSpatialRelationEncoder')
    spa_enc = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=64, coord_dim=2, frequency_num=16, max_radius=5400000, min_radius=50, freq_init='geometric', ffn=ffn, device='cuda')
    pos_enc = ExtentPositionEncoder(spa_enc_type='theory', id2geo=read_id2geo('./Data/id2geo.json'), id2extent=None, spa_enc=spa_enc, graph=graph, spa_enc_embed_norm=False, device='cuda')
    enc = NodeEncoder(feat_enc, pos_enc, agg_type='concat')

    print('Creating Projection Operator..')
    # decoder-projection
    dec =BilinearBlockDiagMetapathDecoder(graph.relations, dims=model_out_dims, feat_dims=out_dims, spa_embed_dim=64)

    print('Creating Intersection Operator..')
    # intersection-attention
    inter_dec = SimpleSetIntersection(agg_func=torch.mean)
    inter_attn = IntersectConcatAttention(query_dims=model_out_dims, key_dims=model_out_dims, num_attn=1, activation='leakyrelu', f_activation='sigmoid', layernorm=True, use_post_mat=True)

    # model
    enc_dec = QueryEncoderDecoder(graph=graph, enc=enc, path_dec=dec, inter_dec=inter_dec, inter_attn=inter_attn, use_inter_node=args.use_inter_node)
    enc_dec.to('cuda')

    print('Loading model..')
    pth_files = [f for f in os.listdir('./model_dir/dbgeo_test/') if f.endswith('.pth')]
    enc_dec.load_state_dict(torch.load('./model_dir/dbgeo_test/' + pth_files[0]))

    print('Running test..')
    perc = eval_perc_queries({formula: [query]}, enc_dec, batch_size=1)

    print('Result: ' + str(perc))
