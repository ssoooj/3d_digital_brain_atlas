import os
import json

def check_ontology(ontology: dict):
    all_ids = []
    roots = ontology['msg']
    print(f'number of roots: {len(roots)}')

    for root in roots:
        nodes_to_be_visited = [root]
        while nodes_to_be_visited:
            node = nodes_to_be_visited.pop()
            all_ids.append(node['id'])
            for child_node in node['children']:
                assert child_node['parent_structure_id'] == node['id'], f"wrong node with id {child_node['id']}"
                print(f"visiting node {child_node['id']} from parent node {node['id']}")
                nodes_to_be_visited.append(child_node)

    print(f'total {len(all_ids)} regions: {all_ids}')


def check_ontology_v2(ontology: dict):
    all_ids = []
    roots = ontology['msg']
    print(f'number of roots: {len(roots)}')

    def check_tree(tree_root: dict):
        all_ids_in_tree = [tree_root['id']]
        if len(tree_root['children']) > 0:
            for child_tree in tree_root['children']:
                assert child_tree['parent_structure_id'] == tree_root['id'], f"wrong node with id {child_tree['id']}"
                print(f"visiting node {child_tree['id']} from parent node {tree_root['id']}")
                all_ids_in_tree.extend(check_tree(child_tree))
        return all_ids_in_tree

    for root in roots:
        all_ids.extend(check_tree(root))

    print(f'total {len(all_ids)} regions: {all_ids}')


def append_children_regions_to_parent_region(parent_region: dict, children_regions: list):
    parent_region['children'].extend(children_regions)
    for child_region in children_regions:
        child_region['parent_structure_id'] = parent_region['id']


def build_lemur_ontology():
    regions = {}

    regions['root'] = {
        "id": 997,
        "atlas_id": -1,
        "ontology_id": 1,
        "acronym": "root",
        "name": "root",
        "color_hex_triplet": "FFFFFF",
        "graph_order": 0,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": "null",
        "children": [],
    }
    regions['grey'] = {
        "id": 8,
        "atlas_id": 0,
        "ontology_id": 1,
        "acronym": "grey",
        "name": "Basic cell groups and regions",
        "color_hex_triplet": "BFDAE3",
        "graph_order": 1,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['CH'] = {
        "id": 567,
        "atlas_id": 70,
        "ontology_id": 1,
        "acronym": "CH",
        "name": "Cerebrum",
        "color_hex_triplet": "7E605D",
        "graph_order": 2,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['CTX'] = {
        "id": 688,
        "atlas_id": 85,
        "ontology_id": 1,
        "acronym": "CTX",
        "name": "Cerebral cortex",
        "color_hex_triplet": "FA7404",
        "graph_order": 3,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Isocortex'] = {
        "id": 315,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "Isocortex",
        "name": "Isocortex",
        "color_hex_triplet": "FACD04",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L1'] = {
        "id": 3159,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L1",
        "name": "Layer1",
        "color_hex_triplet": "FF99CC",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L2&3'] = {
        "id": 3158,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L2&3",
        "name": "Layer2&3",
        "color_hex_triplet": "FF66CC",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L4'] = {
        "id": 3157,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L4",
        "name": "Layer4",
        "color_hex_triplet": "FF33CC",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L5'] = {
        "id": 3156,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L5",
        "name": "Layer5",
        "color_hex_triplet": "FF00CC",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L6'] = {
        "id": 3155,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L6",
        "name": "Layer6",
        "color_hex_triplet": "990099",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    # regions['L6a'] = {
    #     "id": 3154,
    #     "atlas_id": 746,
    #     "ontology_id": 1,
    #     "acronym": "L6a",
    #     "name": "Layer6a",
    #     "color_hex_triplet": "663399",
    #     "graph_order": 5,
    #     "st_level": "null",
    #     "hemisphere_id": 3,
    #     "parent_structure_id": None,
    #     "children": []
    # }
    # regions['L6b'] = {
    #     "id": 3153,
    #     "atlas_id": 746,
    #     "ontology_id": 1,
    #     "acronym": "L6b",
    #     "name": "Layer6b",
    #     "color_hex_triplet": "660099",
    #     "graph_order": 5,
    #     "st_level": "null",
    #     "hemisphere_id": 3,
    #     "parent_structure_id": None,
    # #     "children": []
    # }
    regions['R'] = {
        "id": 3111,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "R",
        "name": "Remaining region",
        "color_hex_triplet": "965251",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['HPF'] = {
        "id": 1089,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "HPF",
        "name": "Hippocampal formation",
        "color_hex_triplet": "5DFA04",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Er'] = {
        "id": 1929,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "Er",
        "name": "Regio Entorhinalis",
        "color_hex_triplet": "FA8FF3",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['CA'] = {
        "id": 1939,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "CA",
        "name": "Ammon's horn",
        "color_hex_triplet": "EFFB72",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['DG'] = {
        "id": 1949,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "DG",
        "name": "Dentate Gyrus",
        "color_hex_triplet": "FBA672",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['S'] = {
        "id": 1959,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "S",
        "name": "Subiculum",
        "color_hex_triplet": "95FB72",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Prs'] = {
        "id": 1969,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "Prs",
        "name": "Presubiculum",
        "color_hex_triplet": "FB72A2",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['IG'] = {
        "id": 1979,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "IG",
        "name": "Indusium Griseum",
        "color_hex_triplet": "5358FC",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['CNU'] = {
        "id": 623,
        "atlas_id": 77,
        "ontology_id": 1,
        "acronym": "CNU",
        "name": "Cerebral nuclei",
        "color_hex_triplet": "04FAEF",
        "graph_order": 520,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['STR'] = {
        "id": 477,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "STR",
        "name": "Striatum",
        "color_hex_triplet": "04B7FA",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Cd'] = {
        "id": 4779,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "Cd",
        "name": "Caudate Nucleus",
        "color_hex_triplet": "088F93",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Put'] = {
        "id": 4778,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "Put",
        "name": "Putamen",
        "color_hex_triplet": "06985A",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['PAL'] = {
        "id": 803,
        "atlas_id": 241,
        "ontology_id": 1,
        "acronym": "PAL",
        "name": "Pallidum",
        "color_hex_triplet": "0440FA",
        "graph_order": 557,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['GPe'] = {
        "id": 4777,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "GPe",
        "name": "Globus Pallidus, external segment",
        "color_hex_triplet": "F69256",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['GPi'] = {
        "id": 4776,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "GPi",
        "name": "Globus Pallidus, internal segment",
        "color_hex_triplet": "6C5B72",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['BS'] = {
        "id": 343,
        "atlas_id": 42,
        "ontology_id": 1,
        "acronym": "BS",
        "name": "Brain stem",
        "color_hex_triplet": "FF7080",
        "graph_order": 588,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['IB'] = {
        "id": 1129,
        "atlas_id": 140,
        "ontology_id": 1,
        "acronym": "IB",
        "name": "Interbrain",
        "color_hex_triplet": "FF7080",
        "graph_order": 589,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['TH'] = {
        "id": 549,
        "atlas_id": 351,
        "ontology_id": 1,
        "acronym": "TH",
        "name": "Thalamus",
        "color_hex_triplet": "5A04FA",
        "graph_order": 590,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['HY'] = {
        "id": 1097,
        "atlas_id": 136,
        "ontology_id": 1,
        "acronym": "HY",
        "name": "Hypothalamus",
        "color_hex_triplet": "E64438",
        "graph_order": 655,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['STN'] = {
        "id": 4775,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "STN",
        "name": "Subthalamic Nucleus",
        "color_hex_triplet": "0F7E32",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['MB'] = {
        "id": 313,
        "atlas_id": 180,
        "ontology_id": 1,
        "acronym": "MB",
        "name": "Midbrain",
        "color_hex_triplet": "AF04FA",
        "graph_order": 740,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['SNc'] = {
        "id": 4774,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "SNc",
        "name": "Substantia nigra, compact part",
        "color_hex_triplet": "FC9D95",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['SNr'] = {
        "id": 4773,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "SNr",
        "name": "Substantia nigra, reticular part",
        "color_hex_triplet": "FE4365",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['HB'] = {
        "id": 1065,
        "atlas_id": 132,
        "ontology_id": 1,
        "acronym": "HB",
        "name": "Hindbrain",
        "color_hex_triplet": "EF04FA",
        "graph_order": 801,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['P'] = {
        "id": 771,
        "atlas_id": 237,
        "ontology_id": 1,
        "acronym": "P",
        "name": "Pons",
        "color_hex_triplet": "FA04AF",
        "graph_order": 802,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['MY'] = {
        "id": 354,
        "atlas_id": 185,
        "ontology_id": 1,
        "acronym": "MY",
        "name": "Medulla",
        "color_hex_triplet": "FA0465",
        "graph_order": 849,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['CB'] = {
        "id": 512,
        "atlas_id": 63,
        "ontology_id": 1,
        "acronym": "CB",
        "name": "Cerebellum",
        "color_hex_triplet": "FA0422",
        "graph_order": 927,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['FT'] = {
            "id": 1009,
        "atlas_id": 691,
        "ontology_id": 1,
        "acronym": "FT",
        "name": "fiber tracts",
        "color_hex_triplet": "A9B00A",
        "graph_order": 1013,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['VS'] = {
        "id": 73,
        "atlas_id": 716,
        "ontology_id": 1,
        "acronym": "VS",
        "name": "ventricular systems",
        "color_hex_triplet": "F58355",
        "graph_order": 1199,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['V1'] = {
        "id": 9101,
        "atlas_id": 716,
        "ontology_id": 1,
        "acronym": "V1",
        "name": "lateral ventricle",
        "color_hex_triplet": "046F12",
        "graph_order": 1199,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['V3'] = {
        "id": 9102,
        "atlas_id": 716,
        "ontology_id": 1,
        "acronym": "V3",
        "name": "third ventricle",
        "color_hex_triplet": "97FEE8",
        "graph_order": 1199,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['V4'] = {
        "id": 9103,
        "atlas_id": 716,
        "ontology_id": 1,
        "acronym": "V4",
        "name": "fourth ventricle",
        "color_hex_triplet": "FD79AB",
        "graph_order": 1199,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    # ....

    parent_to_children_map = {
        'root' : ['grey','FT', 'VS'],
        'grey' : ['CH', 'BS', 'CB'],
        'CH' : ['CTX', 'HPF', 'CNU'],
        'CTX' : ['Isocortex', 'R'],
        'Isocortex' : ['L1', 'L2&3', 'L4', 'L5', 'L6', 'L6a', 'L6b'],
        'HPF' : ['Er','CA', 'DG', 'S', 'Prs', 'IG'],
        'CNU' : ['STR', 'PAL'],
        'STR' : ['Cd', 'Put'],
        'PAL' : ['GPe', 'GPi'],
        'BS' : ['IB', 'MB', 'HB'],
        'IB' : ['TH', 'HY'],
        'HY' : ['STN'],
        'MB' : ['SNc', 'SNr'],
        'HB' : ['P', 'MY'],
        'CB' : [],
        'FT' : [],
        'VS' : ['V1','V3','V4']
    }

    for parent, children in parent_to_children_map.items():
        append_children_regions_to_parent_region(regions[parent],
                                                 [regions[child] for child in children])

    res = {
        "success": "true",
        "id": 0,
        "start_row": 0,
        "num_rows": 1,
        "total_rows": 1,
        "msg": [regions['root']]
    }
    return res


if __name__ == "__main__":
    ontology_copy = os.path.join(os.path.expanduser('~'), 'Downloads', 'lemur_atlas_ontology_v3.json')

    with open(ontology_copy, 'w') as outfile:
        json.dump(build_lemur_ontology(), outfile, indent = 4)

    with open(ontology_copy, 'r') as json_file:
        m_ontology = json.load(json_file)
        check_ontology(m_ontology)
        check_ontology_v2(m_ontology)
