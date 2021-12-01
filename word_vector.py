import gensim


corpus_file = 'wiki_corpus.txt'
model_file = 'wiki_model.model'
wv_file = 'wiki_model.wv'


class WordVector:
    def __init__(self, wv_file):
        self.wv = gensim.models.KeyedVectors.load(wv_file)
    
    def get_vector(self, word):
        return self.wv[word]

    def find_similar(self, vector):
        return self.wv.most_similar(positive = [vector])  # top 10 most similar words


if __name__ == '__main__':

    # generate corpus file
    wiki = gensim.corpora.WikiCorpus('wikitest.bz2', dictionary = {}, lower = True)
    output = open(corpus_file, 'w', encoding = 'utf-8')
    for text in wiki.get_texts():
        output.write(' '.join(text) + '\n')
    output.close()

    # train word2vec model
    model = gensim.models.Word2Vec(sentences = gensim.models.word2vec.LineSentence(corpus_file), vector_size = 500, window = 10)
    model.save(model_file)
    model.wv.save(wv_file)
