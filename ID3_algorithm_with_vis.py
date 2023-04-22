"""
Author : "Jiho Choi" <choi.scholarg@gmail.com>
Description :
    This code is for ID3 algorithm with visualization.
    This code is based on the following pseudocode.

    ### Algorithm pseudo code ###
        buildTree(dataset, Output)
        if all output values are the same in dataset
            return a leaf node saying "predict this unique output value"

        elif all input attributes in dataset are the same
            return a leaf node saying "predict the majority output"

        else find attribute X_n with maximum information gain
            if X_n has n distinct values
                create a non-leaf node with n children
                for each child of X_n
                    BuildTree(DS_i, Output)
                    DS_i is a set of records in dataset where X_n = ith value of X_n

Usage :
let data as,
test_data_2 = np.array([
        ['less', 'Adequate', 'Y', 'False'],
        ['more', 'Huge', 'Y', 'False'],
        ['more', 'Small', 'Y', 'True'],
        ['less', 'Small', 'N', 'True'],
        ['less', 'Adequate', 'N', 'True'],
        ['more', 'Huge', 'N', 'False'],
        ['less', 'Adequate', 'Y', 'False'],
        ['more', 'Huge', 'N', 'False'],
        ['more', 'Small', 'N', 'True'],
        ['less', 'Adequate', 'N', 'True'],
    ])
    test_data_2_feature = ['num_student', 'load', 'room_availability', 'cancelled']

then run the following code, the result will be png file that describe decision tree.

"""

import numpy as np
from graphviz import Digraph


class DecisionNode:
    def __init__(self, is_leaf_node=None, feature=None, IG=None, feature_value=None, prediction=None, train_data=None):
        self.is_leaf_node = is_leaf_node
        self.feature = feature
        self.IG = IG
        self.feature_value = feature_value
        self.prediction = prediction
        self.children = []
        self.train_data = train_data

    def add_child(self, feature_value, child):
        child.feature_value = feature_value
        self.children.append(child)


def visualize_tree(tree: DecisionNode, feature_names):
    dot = Digraph()

    def add_node(tree: DecisionNode, parent: DecisionNode = None):
        node_id = id(tree)

        if tree.feature is not None:
            # 분기가 되는 경우
            label = f'feature= {feature_names[tree.feature]}\nIG= {tree.IG:.3f}'
        else:
            # leaf node 인경우, pred 출력 필요 결과가 되는 경우
            label = f'pred= {tree.prediction}\n#train_data = {tree.train_data}'
        dot.node(str(node_id), label=label)

        if len(tree.children) != 0:
            for child in tree.children:
                child_id = add_node(child, tree)
                dot.edge(str(node_id), str(child_id), label=str(child.feature_value))
        return node_id

    add_node(tree)
    return dot


def entropy(data):
    # 이부분 현재 data에 대하여, y label을 기준으로 entropy를 계산해주는 code이다.

    target_col = data[:, -1]
    # 각각의 label이 어떤 것이고 몇개씩인지 count 해주는 부분
    unique, counts = np.unique(target_col, return_counts=True)
    probs = counts / counts.sum()
    entropy = np.sum(probs * -np.log2(probs))
    return entropy


def information_gain(data, feature_col_num):
    """
    이부분은 data에서 어떠한 column을 기준으로, information gain을
    작성할 것인가에 대한 code이다.

    """
    parent_entropy = entropy(data)
    feature = data[:, feature_col_num]
    feature_unique, feature_count = np.unique(feature, return_counts=True)
    child_probs = feature_count / feature_count.sum()

    # list comprehension을 하면 자동으로 numpy에서 vectorize하게 된다.
    child_data_list = [data[data[:, feature_col_num] == feat] for feat in feature_unique]
    child_entropies = np.array([entropy(child_data) for child_data in child_data_list])
    child_entropy = np.sum(child_probs * child_entropies)

    return parent_entropy - child_entropy


def best_feature(data):
    # features + target을 concatentate 했기 때문이다.
    features = data.shape[1] - 1
    info_gains = [information_gain(data, feature_col_num) for feature_col_num in range(features)]

    max_idx = np.argmax(info_gains)
    return max_idx, info_gains[max_idx], info_gains


# 이 부분은 공부좀 해야겠다... recursive tree build는 어렵다..
# information gain 부분도 추가하면 좋을 것 같다.
def build_tree(data):
    target_col = data[:, -1]

    # y label이 하나만 있는 경우, y label을 반환한다.
    # 다만 여기서, 각 label 별 count를 한다면 좋을 것 같다.
    if len(np.unique(target_col)) <= 1:
        label_unique, label_count = np.unique(target_col, return_counts=True)
        return DecisionNode(is_leaf_node=True, prediction=label_unique[0], train_data=int(label_count[0]))

    # 더 이상 구분할 수 있는 feature가 남아있지 않은 경우이다. why? # features + target을 concatentate 했기 때문이다.
    if data.shape[1] == 1:
        # np.unique의 전달인자에 따라서 단순히 list만 또는 count까지 같이 return해준다.
        # 이때 전체 data의 수는 data.shape[0]에 해당한다.
        return DecisionNode(is_leaf_node=True, prediction=np.unique(target_col)[0], train_data=int(data.shape[0]))

    # 위와 같은 경우가 아니라면, recursive tree generation이 필요하다.
    best_feature_idx, best_feature_IG, info_gains = best_feature(data)

    # 가장 최 상단 tree의 구축, 및 recursive 구축
    # feature name은 visualize할 때 사용하도록 한다.
    tree = DecisionNode(is_leaf_node=False, feature=best_feature_idx, IG=best_feature_IG)

    # 해당 feature를 기반으로 branching을 하기위한 data를 추출하기 위하여,
    best_feature_values = np.unique(data[:, best_feature_idx])
    # 여기가 recursive하게 tree를 구축한다.
    for each_value in best_feature_values:
        # recursive tree 구축
        sub_data = data[data[:, best_feature_idx] == each_value]
        sub_tree = build_tree(sub_data)
        tree.add_child(each_value, sub_tree)
        # [best_feature_name][each_value] = sub_tree
    return tree


if __name__ == "__main__":
    test_data_1 = np.array([
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ])
    test_data_1_feature = ['Outlook', 'Temperature', 'Humidity', 'Wind']

    test_data_2 = np.array([
        ['less', 'Adequate', 'Y', 'False'],
        ['more', 'Huge', 'Y', 'False'],
        ['more', 'Small', 'Y', 'True'],
        ['less', 'Small', 'N', 'True'],
        ['less', 'Adequate', 'N', 'True'],
        ['more', 'Huge', 'N', 'False'],
        ['less', 'Adequate', 'Y', 'False'],
        ['more', 'Huge', 'N', 'False'],
        ['more', 'Small', 'N', 'True'],
        ['less', 'Adequate', 'N', 'True'],
    ])
    test_data_2_feature = ['num_student', 'load', 'room_availability', 'cancelled']

    result_decision_tree_obj = build_tree(test_data_2)
    result_decision_tree_graph = visualize_tree(result_decision_tree_obj, test_data_2_feature)

    # save result graphiz object as png file.
    result_decision_tree_graph.render('result_decision_tree_graph', format='png')
