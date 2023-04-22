from graphviz import Digraph

def visualize_tree(tree, feature_names):
    dot = Digraph()

    def add_node(tree, parent=None):
        node_id = id(tree)
        if 'feature' in tree:
            label = feature_names[tree['feature']]
        else:
            label = str(tree['class'])
        if not parent:
            dot.node(str(node_id), label=label)
        else:
            dot.node(str(node_id), label='{} = {}'.format(feature_names[parent['feature']], parent['value']))
        if 'branches' in tree:
            for b in tree['branches']:
                if 'subtree' in b:
                    child_id = add_node(b['subtree'], parent={'feature': tree['feature'], 'value': b['value']})
                    dot.edge(str(node_id), str(child_id), label=str(b['value']))
                else:
                    child_id = id(b)
                    dot.node(str(child_id), label=str(b['value']))
                    dot.edge(str(node_id), str(child_id), label=str(b['value']))
        return node_id

    add_node(tree)
    return dot

if __name__ == "__main__":
    tree = {'feature': 0, 'branches': [{'value': 0, 'subtree': {'feature': 1,
                                                                'branches': [{'value': 1, 'subtree': {'class': 0}},
                                                                             {'value': 2, 'subtree': {'class': 1}}]}},
                                       {'value': 1, 'subtree': {'class': 1}}, {'value': 2, 'subtree': {'class': 0}},
                                       {'value': 3, 'subtree': {'class': 0}}, {'value': 4, 'subtree': {'class': 1}}]}
    feature_names = ['feature1', 'feature2']

    visualize_tree(tree, feature_names).view()
    test = 1