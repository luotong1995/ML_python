import gensim
import os

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)
input = [['first', 'sentence']]
# model.wv.most_similar()
print(model.wv['first'])


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
    

sentences = MySentences('/some/directory')  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)