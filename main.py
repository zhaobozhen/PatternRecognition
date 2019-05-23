import numpy as np
from matplotlib import pyplot
from collections import Counter


# k-Nearest Neighbor Algorithm
def k_nearest_neighbors(data, predict, k=1):

    # 予測点から各点までの距離を計算
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    sorted_distances = [i[1] for i in sorted(distances)]
    top_nearest = sorted_distances[:k]

    # print(top_nearest)  # ['red','black','red']
    group_res = Counter(top_nearest).most_common(1)[0][0]
    # confidencesはこの分類の確実性の度合い
    confidence = Counter(top_nearest).most_common(1)[0][1] * 1.0 / k
    return group_res, confidence


def load_data(filename):
    data = []
    with open(filename) as txtdata:
        lines = txtdata.readlines()
        for line in lines:
            tempdata = line.strip().split(',')  # スペースとコンマを削除
            tempdata = list(map(float, tempdata))
            data.append(tempdata)

    return data


if __name__ == '__main__':
    file = './data/data1-A.txt'
    data_a = load_data(file)

    file = './data/data1-B.txt'
    data_b = load_data(file)

    file = './data/data1-X.txt'
    predict = load_data(file)

    dataset = {'Data_A': data_a, 'Data_B': data_b}
    colors = {'Data_A': 'black', 'Data_B': 'red'}

    for i in dataset:
        for ii in dataset[i]:
            pyplot.scatter(ii[0], ii[1], s=20, color=colors[i])

    for i in range(0, len(predict)):
        which_group, m_confidence = k_nearest_neighbors(dataset, predict[i], k=100)
        print('Data_X', i+1, 'belongs to', which_group, ', confidence:', m_confidence)
        pyplot.scatter(predict[i][0], predict[i][1], s=100, color='blue')

    pyplot.show()
