import json
import re

import pandas as pd
import os
from rdflib import Graph, Literal, RDF, URIRef
import MinimumBoundingBox.MinimumBoundingBox as mbb


class MyDict(dict):
    def __missing__(self, key):
        return key


def relation_inverse(relation):
    return relation.replace('>', '_inverse>') if '_inverse>' not in relation else relation.replace('_inverse>', '>')


def transform(data):
    data = data.replace('kr.di.uoa.gr', 'kr_di_uoa_gr', regex=True)
    data = data.replace('www.opengis.net', 'www_opengis_net', regex=True)
    data = data.replace('knowledge.org', 'knowledge_org', regex=True)
    data = data.replace('St\.', 'St', regex=True)
    return data


def read_triples(triples_path, change_data=True):
    print(f'reading {triples_path}')
    data = pd.read_csv(triples_path, sep=" ", header=0)
    data.columns = ['head', 'relation', 'tail', '.']
    data.drop(columns=['.'])
    if change_data:
        data = transform(data)
    return data


def read_geo_classes(geo_classes_path, change_data=True):
    geometry2polygon = {}
    entity2type = {}
    entities_dict = {}
    doubles_dict = {}

    for file in os.listdir(geo_classes_path):
        g = Graph()
        print(f'reading {file}')
        g.parse(geo_classes_path+file, format='ttl')

        for geometry, polygon in g.subject_objects(predicate=URIRef('http://www.opengis.net/ont/geosparql#asWKT')):
            geometry2polygon[geometry] = polygon

        for entity, type in g.subject_objects(predicate=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')):
            if entity in entity2type:  # If we have this entity from another dataset keep the old type
                continue
            entity2type[entity] = type

        for entity, geometry in g.subject_objects(predicate=URIRef('http://www.opengis.net/ont/geosparql#hasGeometry')):
            if entity in entities_dict:
                doubles_dict[entity] = [entity, geometry]
                continue
            entities_dict[entity] = [entity, entity2type[entity], geometry, geometry2polygon[geometry]]

    entities_double = pd.DataFrame.from_dict(doubles_dict, orient='index',
                                      columns=['entity', 'geometry']).reset_index(
        drop=True).reset_index().rename(columns={'index': 'id'})

    entities = pd.DataFrame.from_dict(entities_dict, orient='index',
                                      columns=['entity', 'type', 'geometry', 'polygon']).reset_index(
        drop=True).reset_index().rename(columns={'index': 'id'})

    if change_data:
        entities = transform(entities)
        entities_double = transform(entities_double)

    return entities, entities_double


def unite_files(triples_path, geo_classes_path, out_path):
    data = pd.DataFrame()
    # Join all the triples in one file
    for root, directories, files in os.walk(triples_path):
        for name in files:
            data = data.append(read_triples(os.path.join(root, name)))

    data.to_csv(out_path + 'united_triples.nt', sep=' ', index=False)

    # Join all the attributes and geometries of the entities in one file / OUT
    # geo_classes_file = open(out_path + 'geo_classes.ttl', 'a')
    # for file in os.listdir(geo_classes_path):
    #     with open(geo_classes_path + file) as f:
    #         data = f.read()
    #         geo_classes_file.write(data)

    return out_path + 'united_triples.nt'  # , out_path + 'geo_classes.ttl'


def make_id_files(triples_path, geo_classes_path, out_path):
    entities_attributes, entities_attributes_doubles = read_geo_classes(geo_classes_path)
    entities_attributes[['entity', 'type', 'geometry']] = entities_attributes[['entity', 'type', 'geometry']].apply(lambda x: '<'+x+'>')
    entities_attributes_doubles[['entity', 'geometry']] = entities_attributes_doubles[['entity', 'geometry']].apply(lambda x: '<'+x+'>')
    triples = read_triples(triples_path)

    # # Create entity to id file
    # print('Create entity to id file..')
    # entities = entities_attributes[['entity', 'id']]
    # entities.to_csv(out_path + 'entity2id.txt', index=False, sep=' ')
    #
    # # Create relation and inverses to id
    # print('Create relation and inverses to id..')
    # relations = triples['relation'].drop_duplicates().reset_index(drop=True).reset_index().rename(
    #     columns={'index': 'id'})
    # relations = relations[['relation', 'id']]
    #
    # rel = list(relations['relation'])
    # for r in rel:
    #     relations = relations.append({'relation': relation_inverse(r), 'id': len(relations)}, ignore_index=True)
    # relations.to_csv(out_path + 'relation2id.txt', index=False, sep=' ')
    #
    # # Create relation to inverse
    # print('Create relation to inverse..')
    # relation2inverse = {}
    # for relation in relations['relation']:
    #     relation2inverse[relation] = relation_inverse(relation)
    #
    # with open(out_path + 'relation2inverse.json', 'w') as file:
    #     json.dump(relation2inverse, file)
    #
    # # Create relation id to inverse id
    # print('Create relation id to inverse id..')
    # relation_id2inverse_id = {}
    # for i, [relation, id] in relations.iterrows():
    #     relation_id2inverse_id[id] = relations[relations['relation'] == relation_inverse(relation)]['id'].item()
    #
    # with open(out_path + 'rid2inverse.json', 'w') as file:
    #     json.dump(relation_id2inverse_id, file)

    # Change geometries in triples to entities
    # print('Change geometries in triples to entities..')
    # geometry2entity = MyDict(zip(entities_attributes['geometry'].append(entities_attributes_doubles['geometry']), entities_attributes['entity'].append(entities_attributes_doubles['entity'])))
    # triples['head'] = triples['head'].map(geometry2entity)
    # triples['tail'] = triples['tail'].map(geometry2entity)
    # triples.to_csv(triples_path, sep=' ', index=False)

    # Create id to geometries
    print('Create id to geometries..')
    id2geometry = dict(zip(entities_attributes['id'], entities_attributes['polygon']))
    for id in id2geometry.keys():
        print(f'Minimum bounding box for Geometry with id {id}')
        # polygon = [coord_pair.lstrip().split(' ') for coord_pair in
        #            id2geometry[id].split('((')[1].split('))')[0].replace('(', '').replace(')', '').split(
        #                ',')]
        polygon = [[float(n) for n in s.split(' ')] for s in re.findall('\-?\d*\.?\d+\s\-?\d*\.?\d+', id2geometry[id])]  # Take only the pairs of coordinates
        # polygon = [[float(coord) for coord in pair] for pair in polygon]  # Convert strings to floats
        id2geometry[id] = mbb.minimum_bounding_box(polygon)

    with open(out_path + 'id2geo.json', 'w') as file:
        json.dump(id2geometry, file)

    # Create entity to type
    print('Create entity to type..')
    entity2type = entities_attributes[['entity', 'type']]
    entity2type.to_json(out_path + 'entity2type.json', index=False)

    return
