from ci import SignificanceTest
from nltk.translate.bleu_score import corpus_bleu
from pdb import set_trace

def bleu_score(corpus1, corpus2):
    return corpus_bleu(corpus1, corpus2)

def dummy(cor1, cor2):
    return 1.0

if __name__ == "__main__":
    labels = []
    preds1 = []
    preds2 = []
    with open("decode_basic_test.txt", 'r') as f1, open("decode_test_best.txt", 'r') as f2, open("test.de-en.en.wmixerprep", 'r') as f:
        for line1, line2, line in zip(f1, f2, f):
            line = line.strip().split()
            line1 = line1.strip().split()
            line2 = line2.strip().split()
            preds1.append(line1)
            preds2.append(line2)
            labels.append(line)

    labels = list(map(lambda x: [x], labels)) 
    # SignificanceTest(preds1, labels, corpus_bleu, pred2=preds2, type_of_ci='paired_percentile', num_bootstrap=1000)
