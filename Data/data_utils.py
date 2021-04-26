import pandas as pd
import json
import torch
import numpy as np
from scipy.spatial import ConvexHull
from graph import Graph
import matplotlib.pyplot as plt
import MinimumBoundingBox.MinimumBoundingBox as mbb


def read_triples(in_path):
    data = pd.read_csv(in_path, sep=" ")
    data.columns = ['head', 'relation', 'tail', '.']
    return data


def read_custom_triples(custom_triples_path):
    custom_triples = pd.read_csv(custom_triples_path, sep='|')
    custom_triples.columns = ['head', 'rel', 'tail']
    return custom_triples


def find_class(classes_path, entity):
    with open(classes_path) as json_file:
        data = json.load(json_file)
        return data[entity]


def find_id(file_path, row):
    data = pd.read_csv(file_path, sep=" ")
    data.columns = ['data', 'id']
    return data[data.data == row].id


def find_node_maps(classes_path, entitiesID_path):
    entity_ids = pd.read_csv(entitiesID_path, sep=" ")
    entity_ids.columns = ['entity', 'id']

    with open(classes_path) as json_file:
        types = json.load(json_file)

        # Create {type : {entity_id : local_entity_id}}
        node_maps = dict()
        for entity in types:

            entity_type = types[entity]
            if entity_type not in node_maps:
                node_maps[entity_type] = dict()

            entity_id = entity_ids[entity_ids.entity == entity].id.values[0]
            if entity_id not in node_maps[entity_type]:
                node_maps[entity_type][entity_id] = -1

        # Set local entity ids for each type
        for e_type in node_maps:
            for local_entity_id, entity_id in enumerate(node_maps[e_type]):
                node_maps[e_type][entity_id] = local_entity_id

        return node_maps


def make_kr_only_topological(in_path, out_path):
    data = read_triples(in_path)
    data = data[(data['head'].str.contains("kr.di.uoa.gr")) & (data['tail'].str.contains("kr.di.uoa.gr"))]
    data.to_csv(out_path + 'krOnly_OS_topological.nt', index=False, sep=" ")
    return out_path + 'krOnly_OS_topological.nt'


def make_entity2id(in_path, out_path):
    data = read_triples(in_path)
    s = set(set(data['head']).union(set(data['tail'])))
    df = pd.DataFrame([[d, i] for d, i in zip(s, range(len(s)))])
    df.to_csv(out_path + 'entity2id.txt', index=False, sep=" ")
    return out_path + 'entity2id.txt'


def make_relation2id(in_path, out_path):
    data = read_triples(in_path)
    s = set(data['relation'])
    df = pd.DataFrame([d, i] for d, i in zip(s, range(len(s))))
    df.to_csv(out_path + 'relation2id.txt', index=False, sep=" ")
    return out_path + 'relation2id.txt'


def make_id2geo(in_path, out_path):

    # Read geometries
    polygons = dict()
    mbbs = dict()
    with open(in_path) as file:
        file = [li.strip() for li in file if li.strip()]
        last_id = -1
        print('Making Bounding Boxes from Polygons..')
        for line in file:
            if 'POLYGON' in line:
                temp = [coord_pair.split(' ') for coord_pair in line.split('((')[1].split('))')[0].replace('(', '').replace(')', '').split(',')]  # Take only the pairs of coordinates
                polygon = [[float(coord) for coord in pair] for pair in temp]  # Convert strings to floats
                polygons[last_id] = polygon
                mbbs[last_id] = mbb.minimum_bounding_box(polygon)  # Find the minimum bounding box of the polygon

                # Plot the polygon with the bounding box
                temp_polygon = np.array(polygon)
                temp_corners = np.array(mbbs[last_id]['corner_points'])
                plt.scatter(temp_polygon[:, 0], temp_polygon[:, 1], color='green', linewidths=1)
                plt.scatter(temp_corners[0, 0], temp_corners[0, 1], color='red')
                plt.scatter(temp_corners[2, 0], temp_corners[2, 1], color='blue')
                plt.fill(temp_corners[:, 0], temp_corners[:, 1], alpha=0.4, color='yellow')
                plt.axis('equal')
                plt.show()
            elif 'Geometry_OS_' in line:
                last_id = line.split('Geometry_OS_')[1].split('>')[0]  # Take only the id of the geometry following

    print('Writing polygons and Bounding boxes..')
    with open(out_path + 'polygons.json', 'w') as file:
        json.dump(polygons, file)

    with open(out_path + 'id2geo.json', 'w') as file:
        json.dump(polygons, file)

    return out_path + 'id2geo.json'


def read_id2geo(in_path):
    """
    Dictionary in json contains key: geo instance id | value: dictionary with keys:
    area
    length_parallel
    length_orthogonal
    rectangle_center
    unit_vector
    unit_vector_angle
    corner_points

    :param in_path: json filepath
    :return: dictionary of key: id | values: [northeast, southwest] box coordinates
    """

    # Load geo locations
    with open(in_path) as file:
        data = json.load(file)

    id2geo = dict()

    # For each geo location
    for geo_id in data.keys():
        corners = data[geo_id]['corner_points']

        # Find the northeast and the southwest corners
        northeast = corners[2]
        southwest = corners[3]
        # for [x, y] in corners:
        #     if not northeast:
        #         northeast = [x, y]
        #
        #     if not southwest:
        #         southwest = [x, y]
        #
        #     if x >= northeast[0]-() and y >= northeast[1]:
        #         northeast = [x, y]
        #     elif x <= southwest[0] and x <= southwest[1]:
        #         southwest = [x, y]

        id2geo[geo_id] = [northeast, southwest]

    return id2geo


def plot_compare_box_polygon(mbb_path, polygon_path):

    mbb = read_id2geo(mbb_path)

    # Load geo locations
    with open(mbb_path) as file:
        data = json.load(file)

    with open(polygon_path) as file:
        file = [li.strip() for li in file if li.strip()]
        last_id = -1
        for line in file:
            if 'POLYGON' in line:
                temp = [coord_pair.split(' ') for coord_pair in line.split('((')[1].split('))')[0].replace('(', '').replace(')', '').split(',')]  # Take only the pairs of coordinates
                polygon = np.array([[float(coord) for coord in pair] for pair in temp])  # Convert strings to floats
                mbb_corners = np.array(mbb[last_id])
                corners = np.array(data[last_id]['corner_points'])

                plt.scatter(polygon[:, 0], polygon[:, 1], color='red', linewidths=1)
                plt.scatter(mbb_corners[:, 0], mbb_corners[:, 1], color='blue')
                # plt.scatter(corners[:, 0], corners[:, 1], color='yellow')
                plt.axis('equal')
                plt.show()
            elif 'Geometry_OS_' in line:
                last_id = line.split('Geometry_OS_')[1].split('>')[0]  # Take only the id of the geometry following


def make_entity2type(in_path, entitiesID_path, out_path):
    # Read entity ids to check also there for valid entities
    entity_ids = pd.read_csv(entitiesID_path, sep=" ")
    entity_ids.columns = ['entity', 'id']

    # Read types
    ent2type = list()
    with open(in_path) as file:
        file = [li.strip() for li in file if li.strip()]
        head = ['']
        for line in file:
            if line.startswith('a') and head[0] in entity_ids.entity.values:
                ent2type.append([head[0], line.split()[1]])
            else:
                head = [line]

    cols = list(list(zip(*ent2type))[0])
    v = list(list(zip(*ent2type))[1])
    with open(out_path + 'entity2type.json', 'w') as file:
        js = json.dumps([{cols[i]: v[i]} for i in range(len(cols))])
        js = js.replace("[{", "{\n\t")
        js = js.replace("}, {", ",\n\t")
        js = js.replace("}]", "\n}")
        file.write(js)
    return out_path + 'entity2type.json'


def make_custom_triples(triples_path, classes_path, relationsID_path, entitiesID_path, out_path):
    # Read the triples
    data = read_triples(triples_path)

    # Read the relations to id's
    relations = pd.read_csv(relationsID_path, sep=" ")
    relations.columns = ['relation', 'id']

    # Read the entities to id's
    entities = pd.read_csv(entitiesID_path, sep=" ")
    entities.columns = ['entity', 'id']

    # Open the classes json
    with open(classes_path) as json_file:
        classes = json.load(json_file)

        # for each relation find its head and tail classes and head relation tail id's
        custom_triples = list()
        for index, row in data.iterrows():
            head_class = classes[row['head']]
            tail_class = classes[row['tail']]
            relation_id = relations[relations.relation == row['relation']].id.values[0]
            head_id = entities[entities.entity == row['head']].id.values[0]
            tail_id = entities[entities.entity == row['tail']].id.values[0]

            # (head_id, (head_class, relation_id, tail_class), tail_id)
            custom_triple = (head_id, (head_class, relation_id, tail_class), tail_id)

            custom_triples.append(custom_triple)

    df = pd.DataFrame(custom_triples)
    df.to_csv(out_path + 'custom_triples.txt', index=False, sep='|')

    return out_path + 'custom_triples.txt'


def make_graph(custom_triples_path, classes_path, entitiesID_path, embed_dim=10):
    custom_triples = read_custom_triples(custom_triples_path)

    adj_lists = dict()
    relations = dict()
    for custom_triple in custom_triples.iterrows():

        # Separate head, relation and tail
        rel = custom_triple[1].rel.strip("(')").replace("'", "").split(',')  # (head_class, pred_id, tail_class)
        rel[1] = int(rel[1])
        rel = tuple(rel)
        head = int(custom_triple[1][0])  # head_id
        tail = int(custom_triple[1][2])  # tail.id

        # Create Adjacent Lists {(head_class, pred_id, Tail_class) : {head_id : [tail entity id's]}}
        if rel not in adj_lists:
            adj_lists[rel] = dict()

        if head not in adj_lists[rel]:
            adj_lists[rel][head] = list()

        adj_lists[rel][head].append(tail)

        # Create Relations {head_class : [[tail_class, pred_id]]}
        if rel[0] not in relations:
            relations[rel[0]] = list()

        relations[rel[0]].append((rel[2], rel[1]))

    # Create Node Maps {class : {entity_id : local_entity_id}}
    node_maps = find_node_maps(classes_path, entitiesID_path)

    # Delete duplicates from relation lists
    for key in relations:
        relations[key] = set(relations[key])

    # ???
    for m in node_maps:
        node_maps[m][-1] = -1

    # For each type set feature dimension equal to embed_dim
    feature_dims = {m: embed_dim for m in relations}

    # For each type
    feature_modules = dict()
    for e_type in relations:
        # initialize embedding matrix for each type with (num of embeddings = num of ent per type + 1, embed_dim = 10)
        feature_modules[e_type] = torch.nn.Embedding(len(node_maps[e_type]) + 1, embed_dim)

        # define embedding initialization method: normal dist
        feature_modules[e_type].weight.data.normal_(0, 1. / embed_dim)

    def features(nodes, e_type):
        return feature_modules[e_type](torch.autograd.Variable(torch.LongTensor([node_maps[e_type][n] for n in nodes]) + 1))

    graph = Graph(features, feature_dims, relations, adj_lists)

    return graph, feature_modules, node_maps


if __name__ == '__main__':
    triples_path = '../yago2geo_uk/os/OS_topological.nt'
    types_path = '../yago2geo_uk/os/OS_new.ttl'
    geo_path = '../yago2geo_uk/os/OS_extended.ttl'
    data_path = './'

    classes_path = data_path + 'entity2type.json'
    new_triples_path = data_path + 'krOnly_OS_topological.nt'
    relationsID_path = data_path + 'relation2id.txt'
    entitiesID_path = data_path + 'entity2id.txt'
    custom_triples = data_path + 'custom_triples.txt'
    id2geo_path = data_path + 'id2geo.json'

    # Filter the triples to contain kr.di.uoa.gr
    # new_triples_path = make_kr_only_topological(triples_path, data_path)

    # Make the entity2id file (from kr only triples)
    # entitiesID_path = make_entity2id(new_triples_path, data_path)

    # Make the relation2id file
    # relationsID_path = make_relation2id(new_triples_path, data_path)

    # Make the entity2type file
    # classes_path = make_entity2type(types_path, entitiesID_path, data_path)

    # Make the triples
    # custom_triples = make_custom_triples(new_triples_path, classes_path, relationsID_path, entitiesID_path, data_path)

    # Make id2geo file
    id2geo_path = make_id2geo(geo_path, data_path)

    # Get id2geo
    # id2geo = read_id2geo(id2geo_path)

    # Plot Polygons and Bounding Boxes
    plot_compare_box_polygon(id2geo_path, geo_path)

    # Make the graph file
    # make_graph(custom_triples, classes_path, entitiesID_path)
