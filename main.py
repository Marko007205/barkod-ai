import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from surprise import SVD, Reader, Dataset


def read_xlsx(file, sheet_name):
    return pd.read_excel(file, sheet_name=sheet_name)


def xlsx_to_vector(xlsx):
    return xlsx.values.tolist()


def calculate_recall_and_precision(vector1, vector2):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(vector1)):
        if vector1[i] == 1 and vector2[i] == 1:
            true_positives += 1
        elif vector1[i] == 0 and vector2[i] == 1:
            false_positives += 1
        elif vector1[i] == 1 and vector2[i] == 0:
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    return recall, precision


if __name__ == '__main__':
    train_data = read_xlsx('dataset/train.xlsx', 'data')
    train_data.drop_duplicates(inplace=True)
    test_data = read_xlsx('dataset/test.xlsx', 'data')
    test_data.drop_duplicates(inplace=True)

    train_data_vector_list = xlsx_to_vector(train_data[1:].iloc[:, 1:])
    test_data_vector_list = xlsx_to_vector(test_data[1:].iloc[:, 1:])

    test_variable = None
    for count, vec1 in enumerate(test_data_vector_list):
        max_similarity = 0
        most_similar_vec = None
        for vec2 in train_data_vector_list:
            similarity = jaccard_score(vec1[:7], vec2[:7], average='micro')
            if similarity > max_similarity:
                max_similarity = similarity
                if count == len(test_data_vector_list) - 1:
                    test_variable = vec2
                most_similar_vec = vec2
        print("Recommended: ", most_similar_vec[33:])

    print(train_data_vector_list[119][33:])
    print(test_variable[33:])

    recall, precision = calculate_recall_and_precision(train_data_vector_list[119][33:], test_variable[33:])
    print('Recall', recall * 100)
    print('Precision', precision * 100)


# model = SVD()
    # model.fit(train_data)
    # predictions = model.test(test_data)

    # recommendations = {}
    # for uid, iid, _, _ in test_data:
    #     if uid not in recommendations:
    #         recommendations[uid] = []
    #     annexes = train_data.loc[train_data['InsuredName'] == uid]['BusinessClassification'].unique()
    #     for annex in annexes:
    #         pred = model.predict(uid, annex)
    #         recommendations[uid].append((annex, pred.est))
    #     recommendations[uid].sort(key=lambda x: x[1], reverse=True)
    #     recommendations[uid] = [r[0] for r in recommendations[uid][:10]]
    #
    # print(recommendations[test_data[0][0]])
