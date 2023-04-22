import numpy as np

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
    return np.argmax(info_gains)

# 이 부분은 공부좀 해야겠다... recursive tree build는 어렵다..
# information gain 부분도 추가하면 좋을 것 같다.
def build_tree(data, feature_names):
    target_col = data[:, -1]

    # y label이 하나만 있는 경우, desired case
    if len(np.unique(target_col)) <= 1:
        return np.unique(target_col)[0]

    # 더 이상 구분할 수 있는 feature가 남아있지 않은 경우이다. why? # features + target을 concatentate 했기 때문이다.
    if data.shape[1] == 1:
        # np.unique의 전달인자에 따라서 단순히 list만 또는 count까지 같이 return해준다.
        return np.unique(target_col)[np.argmax(np.unique(target_col, return_counts=True)[1])]

    # 위와 같은 경우가 아니라면, recursive tree generation이 필요하다.
    best_feature_idx = best_feature(data)
    best_feature_name = feature_names[best_feature_idx]

    # 가장 최 상단 tree의 구축, 및 recursive 구축
    tree = {best_feature_name: {}}

    best_feature_values = np.unique(data[:, best_feature_idx])
    for each_value in best_feature_values:
        # recursive tree 구축
        sub_data = data[data[:, best_feature_idx] == each_value]
        sub_tree = build_tree(sub_data, feature_names)
        tree[best_feature_name][each_value] = sub_tree
    return tree

if __name__ == "__main__":
    """
    ### Algorithm pseudo code ### 
    
    buildTree(dataset, Output)
    if all output values are the same in dataset
        return a leaf node saying "predict this unique output value"
    elif all input attributes in dataset are the same
        return a leaf node saying "predict the majority output"
    else find attribute X_n with maximum information gain
        if X_n has n distinct values
            create a non-leaf node with n children
            for each children of X_n
                BuildTree(DS_i, Output) 
                DS_i is a set of records in dataset where X_n = ith value of X_n    
    """

    """
    experimental data
    
        data = np.array([
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

    # Sample feature names
    feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    
    """

    data = np.array([
            [0, 1, 1, 0],
            [1, 2, 1, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 2, 0, 0],
            [0, 1, 1, 0],
            [1, 2, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ])
    feature_names = ['num_student', 'load', 'room_availability', 'cancelled']
    result_decision_tree = build_tree(data, feature_names)



    test =1