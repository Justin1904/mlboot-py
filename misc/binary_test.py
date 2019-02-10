from confidence_intervals import bca
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    labels = []
    pred1 = []
    pred2 = []
    with open('binary.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            lab, pre1, pre2 = map(int, line.strip().split(','))
            labels.append(lab)
            pred1.append(pre1)
            pred2.append(pre2)
    full = accuracy_score(pred1, labels)
    print(full)
    a, b, scores = bca(pred1, labels, accuracy_score, 0.95, len(labels), 10000)
    print(a, b)
