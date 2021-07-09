from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pickle

with open("../../data/attrs.pkl", 'rb') as handle:
  text = pickle.load(handle)

model = Word2Vec(text.tolist(), vector_size=4096)
model_2 = Word2Vec(vector_size=4096, min_count=1)
model_2.build_vocab(text)
total_examples = model_2.corpus_count
model_2.train(text.tolist(), total_examples=total_examples, epochs=200)
model.save("../../data/word2vec.model")

X = model_2.wv.vectors
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(list(model_2.wv.index_to_key)):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
#pyplot.savefig("../../data/pca.jpg")