from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pickle


def train_word2vec(text, vector_size=9):
    model = Word2Vec(text.tolist(), vector_size=vector_size)
    model_2 = Word2Vec(vector_size=vector_size, min_count=1)
    model_2.build_vocab(text)
    total_examples = model_2.corpus_count
    model_2.train(text.tolist(), total_examples=total_examples, epochs=200)
    model.save("../data/word2vec{}d.model".format(vector_size))
    #pca(model)

def pca(model):
    X = model.wv.vectors
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(list(model.wv.index_to_key)):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
    pyplot.savefig("../../data/pca.jpg")


if __name__ == "__main__":
    with open("../data/attrs.pkl", 'rb') as handle:
      text = pickle.load(handle)
    train_word2vec(text)
    model = Word2Vec.load("../data/word2vec.model")
    pca(model)