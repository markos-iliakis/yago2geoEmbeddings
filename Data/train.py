from Data.data_creator import make_graph
from trainer import *
import graph


def my_test(trainer):
    # 1260|('<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>', 4, '<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>')|42310
    # <http://kr_di_uoa_gr/yago2geo/resource/osentity_Harston_and_Hauxton_Ward_1260> 1260
    # <http://www_opengis_net/ont/geosparql#sfTouches> 4
    # <http://kr_di_uoa_gr/yago2geo/resource/osentity_The_Shelfords_and_Stapleford_Ward_42310> 42310

    f = graph.Formula('1-chain', (('<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>', '4', '<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>'),))
    query = graph.Query(('1-chain', (1260, ('<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>', '4', '<http://kr_di_uoa_gr/yago2geo/ontology/OS_DistrictWard>'), 42310)))


if __name__ == '__main__':
    # define arguments
    parser = make_args_parser()
    args = parser.parse_args()

    classes_path = args.data_path + 'entity2type.json'
    entitiesID_path = args.data_path + 'entity2id.txt'
    custom_triples = args.data_path + 'custom_triples.txt'
    graph_path = args.data_path + 'testing_graph.pkl'

    print("Loading graph data..")
    graph, feature_modules, node_maps = read_graph(graph_path)
    out_dims = {e_type: args.embed_dim for e_type in graph.relations}

    trainer = Trainer(args, graph, feature_modules, node_maps, out_dims, console=True)

    # load 1-d edge query
    trainer.load_edge_data()

    # load multi edge query
    trainer.load_multi_edge_query_data(load_geo_query=False)

    if args.geo_train:
        # load multi edge geographic query
        trainer.load_multi_edge_query_data(load_geo_query=True)

    # load model
    if args.load_model:
        trainer.load_model()

    print("Training ...")
    # train NN model
    trainer.train()

    my_test(trainer)

    trainer.logger.info("geo_info: {}".format(trainer.args.geo_info))
    trainer.logger.info("lr: {:f}".format(trainer.args.lr))
    trainer.logger.info("freq: {:d}".format(trainer.args.freq))
    trainer.logger.info("max_radius: {:f}".format(trainer.args.max_radius))
    trainer.logger.info("min_radius: {:f}".format(trainer.args.min_radius))
    trainer.logger.info("num_hidden_layer: {:d}".format(trainer.args.num_hidden_layer))
    trainer.logger.info("hidden_dim: {:d}".format(trainer.args.hidden_dim))
    trainer.logger.info("embed_dim: {}".format(trainer.args.embed_dim))
