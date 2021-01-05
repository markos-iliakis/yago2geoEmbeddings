import pandas as pd
from rdflib import Graph
import json


def krOnlyTopological(in_path, out_path):
    data = read_triples(in_path)
    data = data[(data['head'].str.contains("kr.di.uoa.gr")) & (data['tail'].str.contains("kr.di.uoa.gr"))]
    data.to_csv(out_path + 'krOnly_OS_topological.nt', index=False, sep=" ")
    return out_path + 'krOnly_OS_topological.nt'


def read_triples(in_path):
    data = pd.read_csv(in_path, sep=" ")
    data.columns = ['head', 'relation', 'tail', '.']
    return data


def entity2id(in_path, out_path):
    data = read_triples(in_path)
    s = set(set(data['head']).union(set(data['tail'])))
    df = pd.DataFrame([[d, i] for d, i in zip(s, range(len(s)))])
    df.to_csv(out_path + 'entity2id.txt', index=False, sep=" ")
    return out_path + 'entity2id.txt'


def relation2id(in_path, out_path):
    data = read_triples(in_path)
    s = set(data['relation'])
    df = pd.DataFrame([d, i] for d, i in zip(s, range(len(s))))
    df.to_csv(out_path + 'relation2id.txt', index=False, sep=" ")
    return out_path + 'relation2id.txt'


def entity2type(in_path, out_path):
    # Read types
    ent2type = list()
    with open(in_path) as file:
        file = [li.strip() for li in file if li.strip()]
        for line in file:
            if line.startswith('a'):
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


def find_class(classes_path, entity):
    with open(classes_path) as json_file:
        data = json.load(json_file)
        return data[entity]


def find_id(file_path, row):
    data = pd.read_csv(file_path, sep=" ")
    data.columns = ['data', 'id']
    return data[data.data == row].id


def make_triples(triples_path, classes_path, relationsID_path, entitiesID_path, out_path):
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
            relation_id = relations[relations.relation == row['relation']].id
            head_id = entities[entities.entity == row['head']].id
            tail_id = entities[entities.entity == row['tail']].id

            custom_triples.append((head_id, (head_class, relation_id, tail_class), tail_id))

    df = pd.DataFrame(custom_triples)
    df.to_csv(out_path + 'custom_triples.txt', index=False, sep=" ")

    return out_path + 'custom_triples.txt'


if __name__ == '__main__':
    triples_path = './yago2geo_uk/os/OS_topological.nt'
    types_path = './yago2geo_uk/os/OS_new.ttl'
    data_path = './Data/'

    classes_path = data_path + 'entity2type.json'
    new_triples_path = data_path + 'krOnly_OS_topological.nt'
    relationsID_path = data_path + 'relation2id.txt'
    entitiesID_path = data_path + 'entity2id.txt'
    custom_triples = data_path + 'custom_triples.txt'

    # Filter the triples to contain kr.di.uoa.gr
    # new_triples_path = krOnlyTopological(triples_path, data_path)

    # Make the entity2id file (from kr only triples)
    # entitiesID_path = entity2id(new_triples_path, data_path)

    # Make the relation2id file
    # relationsID_path = relation2id(new_triples_path, data_path)

    # Make the entity2type file
    # classes_path = entity2type(types_path, data_path)

    # Make the triples
    custom_triples = make_triples(new_triples_path, classes_path, relationsID_path, entitiesID_path, data_path)

    # Make the graph file
    print('hi')
