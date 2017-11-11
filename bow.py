import sys
import re
import nltk
import csv
import sklearn.feature_extraction.text as txt
from nltk.corpus import stopwords

def clear_text(raw_text):
    letters_only = re.sub("[^a-zA-Z]", "", raw_text)
    lowered_case = letters_only.lower()
    stopwords_portuguese = stopwords.words('portuguese')
    words = lowered_case.split()
    words = [w for w in words if w not in stopwords_portuguese]
    return (" ".join(words))

def read_corpus(filename):
    corpus = []
    with open(filename, mode="r") as input:
        reader = csv.DictReader(input, delimiter=",")
        for row in reader:
            corpus.append(row['tweet'])
    return corpus

def read_labels(filename):
    labels = []
    with open(filename, mode="r") as input:
        reader = csv.DictReader(input, delimiter=",")
        for row in reader:
            labels.append(row['review'])
    return labels

def clear_corpus(corpus):
    clean_corpus = []
    for document in corpus:
        clean_corpus.append(clear_text(document))
    return clean_corpus

def vectorize_frequency(corpus):
    vectorizer = txt.CountVectorizer(analyzer="word", \
            tokenizer=None, \
            preprocessor=None, \
            stop_words=None, \
            max_features=5000)
    corpus_features = vectorizer.fit_transform(corpus)
    return corpus_features.toarray()

def vectorize_binary(corpus):
    vectorizer = txt.CountVectorizer(analyzer="word", \
            tokenizer=None, \
            preprocessor=None, \
            stop_words=None, \
            binary=True, \
            max_features=5000)
    corpus_features = vectorizer.fit_transform(corpus)
    return corpus_features.toarray()

def vectorize_tf_idf(corpus):
    vectorizer = txt.TfidfVectorizer(analyzer="word", \
            tokenizer=None, \
            preprocessor=None, \
            stop_words=None, \
            max_features=5000)
    corpus_features =  vectorizer.fit_transform(corpus)
    return corpus_features.toarray()

def write_to_csv(corpus_features, labels, filename):
    fieldnames = ['x' + str(i) for i in range(corpus_features.shape[1])]
    fieldnames.append('label')
    with open(filename, mode="w") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        for features, label in zip(corpus_features, labels):
            row = {}
            for i in range(corpus_features.shape[1]):
                row['x' + str(i)] = features[i]
            row['label'] = label
            writer.writerow(row)

def main(filename):
    if filename is None:
        raise RuntimeError("Por favor, forneca um nome de arquivo")
        sys.exit(1)
    corpus = read_corpus(filename)
    labels = read_labels(filename)
    clean_corpus = clear_corpus(corpus)
    tfs = vectorize_tf_idf(clean_corpus)
    freqs = vectorize_frequency(clean_corpus)
    binaries = vectorize_binary(clean_corpus)
    write_to_csv(freqs, labels, "output_freq.csv")
    write_to_csv(tfs, labels, "output_tf.csv")
    write_to_csv(binaries, labels, "output_binary.csv")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python bow.py <nome-do-arquivo>")
        sys.exit(1)
    main(sys.argv[1])
