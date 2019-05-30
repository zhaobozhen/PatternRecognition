import math


# 平均値と分散を計算
def calculate_mean_and_variance(dataset):
    x1_sum = 0.0
    x2_sum = 0.0

    for i in dataset:  # 個々の属性の合計を計算
        x1_sum += float(i[0])
        x2_sum += float(i[1])

    # 各属性のサンプルの平均値を計算
    mean_x1 = x1_sum / 500
    mean_x2 = x2_sum / 500

    k11 = 0.0
    k12 = 0.0
    k21 = 0.0
    k22 = 0.0
    # 各属性のサンプルの分散を計算
    for i in dataset:
        k11 += (float(i[0]) - mean_x1) ** 2
        k12 += (float(i[0]) - mean_x1) * (float(i[1]) - mean_x2)
        k21 += (float(i[0]) - mean_x1) * (float(i[1]) - mean_x2)
        k22 += (float(i[1]) - mean_x2) ** 2

    variance_x1 = k11 / 500
    variance_x2 = k22 / 500
    variance_x3 = k12 / 500
    variance_x4 = k21 / 500

    return mean_x1, mean_x2, variance_x1, variance_x2, variance_x3, variance_x4


# 各属性の条件付き確率を計算
def calculate_P_xi_c(mean_x1, mean_x2, variance_x1, variance_x2, test_data):
    p_x1_c = (1 / math.sqrt(2 * math.pi)) * math.exp(-(float(test_data[0]) - mean_x1) ** 2 / (2 * variance_x1))
    p_x2_c = (1 / math.sqrt(2 * math.pi)) * math.exp(-(float(test_data[1]) - mean_x2) ** 2 / (2 * variance_x2))

    return p_x1_c, p_x2_c


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
    file = './data/data2-A.txt'
    data_a = load_data(file)

    file = './data/data2-B.txt'
    data_b = load_data(file)

    file = './data/data2-X.txt'
    predict = load_data(file)

    data_a_mean_x1, data_a_mean_x2, data_a_variance_x1, data_a_variance_x2, data_a_variance_x3, data_a_variance_x4 = calculate_mean_and_variance(
        data_a)
    data_b_mean_x1, data_b_mean_x2, data_b_variance_x1, data_b_variance_x2, data_b_variance_x3, data_b_variance_x4 = calculate_mean_and_variance(
        data_b)

    # 共分散行列を計算
    print('Mean of Data A:', data_a_mean_x1, data_a_mean_x2)
    print('Mean of Data B:', data_b_mean_x1, data_b_mean_x2)
    print('Covariance matrix of Data A:')
    print('[', data_a_variance_x1, data_a_variance_x3, ']')
    print('[', data_a_variance_x4, data_a_variance_x2, ']')
    print('Covariance matrix of Data B:', data_b_variance_x1 + data_b_variance_x2)
    print('[', data_b_variance_x1, data_b_variance_x3, ']')
    print('[', data_b_variance_x4, data_b_variance_x2, ']')

    # 推定事前確率
    p1 = 0.5
    p2 = 0.5
    for x in predict:
        P_x1_data_a, P_x2_data_a = calculate_P_xi_c(data_a_mean_x1, data_a_mean_x2, data_a_variance_x1,
                                                    data_a_variance_x2, x)
        P_x1_data_b, P_x2_data_b = calculate_P_xi_c(data_b_mean_x1, data_b_mean_x2, data_b_variance_x1,
                                                    data_b_variance_x2, x)

        # 事後確率を計算
        P_data_a = p1 * P_x1_data_a * P_x2_data_a
        P_data_b = p2 * P_x1_data_b * P_x2_data_b

        if P_data_a > P_data_b:
            print('Data X', x[0], x[1], 'belongs to Data A.')
        else:
            print('Data X', x[0], x[1], 'belongs to Data B.')