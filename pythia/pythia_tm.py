import io
import copy
import string
import pandas as pd
import numpy as np
import gensim
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans


class TokenizedSentences(object):
    def __init__(self, doclist):
        self.doclist = doclist

    def __iter__(self):
        for doc in self.doclist:
            yield [token for token in doc.split() if len(token) > 1]


class PythiaTM(object):
    """docstring for PythiaTM."""
    def __init__(self, algorithm=None):
        super(PythiaTM, self).__init__()
        self.dictionary = None
        self.tfidf_model = None
        self.model = None

        self.model_feats = list()
        self._prep_opts = list()
        self.algorithm = algorithm

    def train(self, form_data):
        self._prep_opts = form_data.prep_opts.data

        df = pd.read_csv(io.BytesIO(form_data.input_file.data.read()))
        df = self._preprocess(df)
        self._fit(df, form_data)

    def explore(self):
        return self.model_feats

    def predict(self, form_data):
        df = pd.read_csv(io.BytesIO(form_data.input_file.data.read()))
        df = self._preprocess(df, train=False)
        return self._score(df)

    def _score(self, data):
        if self.algorithm == 'kmeans':
            scores = np.zeros(len(data))
            doc_yield = TokenizedSentences(data.values.tolist())
            for i_doc, document in enumerate(doc_yield):
                tf_idf_doc = gensim.matutils.corpus2csc([self.tfidf_model[self.dictionary.doc2bow(document)]],
                                                        num_terms=self.model.cluster_centers_.shape[1]).transpose()
                scores[i_doc] = self.model.predict(tf_idf_doc)
        elif self.algorithm in ('lsi', 'lda'):
            scores = np.zeros((len(data), self.model.num_topics))
            doc_yield = TokenizedSentences(data.values.tolist())
            for i_doc, document in enumerate(doc_yield):
                topic_scores = self.model[self.tfidf_model[self.dictionary.doc2bow(document)]]
                if len(topic_scores) == scores.shape[1]:
                    scores[i_doc, :] = [sc[1] for sc in topic_scores]
        elif self.algorithm == 'w2v':
            scores = self.model.score(TokenizedSentences(data))
        return scores

    def _preprocess(self, data, train=True):
        clean_data = copy.deepcopy(data['Text'])

        # First result: raw data
        if train:
            self.model_feats.append(clean_data.head().values.tolist())

        if 'clean' in self._prep_opts:
            # Remove punctuation and to lowercase
            punct = string.punctuation
            clean_data = clean_data.apply(lambda s: s.translate(s.maketrans(punct, ' '*len(punct)))).str.lower()

        if 'stop' in self._prep_opts:
            # Only english language supported yet
            stop_words = stopwords.words('english')
            clean_data = clean_data.apply(lambda s: ' '.join([word for word in s.split() if word not in stop_words]))

        if 'stem' in self._prep_opts:
            # PorterStemmer used
            ps = PorterStemmer()
            clean_data = clean_data.apply(lambda s: ' '.join([ps.stem(word) for word in s.split()]))

        if 'lemma' in self._prep_opts:
            # PorterStemmer used
            lm = WordNetLemmatizer()
            clean_data = clean_data.apply(lambda s: ' '.join([lm.lemmatize(word) for word in s.split()]))

        # Lowercase
        clean_data = clean_data.str.lower()

        # Next result: Preprocessed data
        if train:
            self.model_feats.append(clean_data.head().values.tolist())

        # Store clean data
        return clean_data

    def _fit(self, data, opts):
        if self.algorithm != 'w2v':
            # TF-IDF
            self.dictionary = gensim.corpora.Dictionary([doc.split() for doc in data.values.tolist()])

            # Filter
            self.dictionary.filter_extremes(no_below=opts.min_df.data, no_above=opts.max_df.data,
                                            keep_n=opts.max_feats.data)

            corpus = list()
            doc_yield = TokenizedSentences(data.values.tolist())
            for document in doc_yield:
                corpus.append(self.dictionary.doc2bow(document))

            self.model_feats.append(self.dictionary)
            self.tfidf_model = gensim.models.TfidfModel(corpus)
            tfidf_matrix = self.tfidf_model[corpus]

            if self.algorithm == 'kmeans':
                corpus_tfidf = gensim.matutils.corpus2csc(tfidf_matrix).transpose()
                self.model = KMeans(n_clusters=opts.n_clusters.data, init='k-means++', max_iter=1000, n_init=1)
                self.model.fit(corpus_tfidf)
                self.model_feats.append(self.model)

            elif self.algorithm == 'lsi':
                self.model = gensim.models.lsimodel.LsiModel(corpus=tfidf_matrix, id2word=self.dictionary,
                                                             num_topics=opts.num_topics.data)
                self.model_feats.append(self.model.print_topics(opts.num_topics.data))

            elif self.algorithm == 'lda':
                self.model = gensim.models.ldamodel.LdaModel(corpus=tfidf_matrix, id2word=self.dictionary,
                                                             num_topics=opts.num_topics.data, update_every=1,
                                                             chunksize=10000, passes=1)
                self.model_feats.append(self.model.print_topics(opts.num_topics.data))

        elif self.algorithm == 'w2v':
            self.model = gensim.models.Word2Vec(sentences=TokenizedSentences(data), size=opts.size.data,
                                                window=opts.window.data, min_count=opts.min_df.data, workers=4,
                                                max_vocab_size=opts.max_vocab.data, hs=1, negative=0)
            self.model_feats.append(self.model)
