"""
Pythia-TM main module.

Holds the main class for training, scoring and exploring
different text mining workflows. It integrates with the
web application that holds this module.
"""
import io
import copy
import pandas as pd
import numpy as np
import gensim
from flask import current_app
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cluster import MiniBatchKMeans


class TokenizedSentences:
    """
    Gensim-style class to provide tokenized sentences that clean
    text before send it to the algorithms.

    Tokens with a single character are removed.
    """
    def __init__(self, doclist):
        self.doclist = doclist

    def __iter__(self):
        for doc in self.doclist:
            yield [token for token in doc.split() if len(token) > 1]


class PythiaTM:
    """
    Main class for handling Pythia-TM models. It stores all objects
    necessary to train, score or explore the models, in order to
    provide the web application the necessary data to operate normally.

    In order to apply different algorithms, PythiaTM uses third-party
    libraries:
    - Scikit-Learn
    - Gensim
    - NLTK
    """
    def __init__(self, algorithm=None):
        super(PythiaTM, self).__init__()
        self.dictionary = None
        self.tfidf_model = None
        self.model = None

        self.model_feats = list()
        self._prep_opts = list()
        self.algorithm = algorithm

    def train(self, form_data):
        """
        Public method to train models. Wrapper over _fit internal method.
        """
        self._prep_opts = form_data.prep_opts.data

        current_app.logger.info('TRAIN | Data acquisition')
        data_df = pd.read_csv(io.BytesIO(form_data.input_file.data.read()))

        current_app.logger.info('TRAIN | Data preprocessing')
        data_df = self._preprocess(data_df, form_data.text_field.data)

        current_app.logger.info('TRAIN | Fitting algorithm(s)')
        self._fit(data_df, form_data)
        current_app.logger.info('TRAIN | Process ended.')

    def explore(self):
        """
        Public method to expose the model features so model can be explored.
        """
        return self.model_feats

    def predict(self, form_data):
        """
        Public method to score new data. Wrapper over _score internal method.
        """
        current_app.logger.info('PREDICT | Data acquisition')
        data_df = pd.read_csv(io.BytesIO(form_data.input_file.data.read()))

        current_app.logger.info('PREDICT | Data preprocessing')
        data_df = self._preprocess(data_df, form_data.text_field.data, train=False)

        current_app.logger.info('PREDICT | Model scoring')
        return self._score(data_df)

    def _score(self, data):
        """
        Internal scoring method.
        """
        if self.algorithm == 'kmeans':
            scores = np.zeros(len(data))
            doc_yield = TokenizedSentences(data.values.tolist())
            for i_doc, document in enumerate(doc_yield):
                # Apply TF-IDF
                tf_idf_doc = gensim.matutils.corpus2csc([self.tfidf_model[self.dictionary.doc2bow(document)]],
                                                        num_terms=self.model.cluster_centers_.shape[1]).transpose()

                # Score KMeans model
                scores[i_doc] = self.model.predict(tf_idf_doc)
        elif self.algorithm in ('lsi', 'lda'):
            scores = np.zeros((len(data), self.model.num_topics))
            doc_yield = TokenizedSentences(data.values.tolist())
            for i_doc, document in enumerate(doc_yield):
                # Apply TF-IDF + LSI/LDA models
                topic_scores = self.model[self.tfidf_model[self.dictionary.doc2bow(document)]]
                if len(topic_scores) == scores.shape[1]:
                    scores[i_doc, :] = [sc[1] for sc in topic_scores]
        elif self.algorithm == 'w2v':
            # Score W2V model
            scores = self.model.score(TokenizedSentences(data))
        return scores

    def _preprocess(self, data, field, train=True):
        """
        Internal data preprocessing method.
        """
        clean_data = copy.deepcopy(data[field])

        # First result: raw data
        if train:
            self.model_feats.append(clean_data.head().values.tolist())

        # Lowercase first
        clean_data = clean_data.str.lower()

        if 'clean' in self._prep_opts:
            # Remove punctuation from gensim module
            current_app.logger.info('Clean text...')
            clean_data = clean_data.apply(gensim.parsing.preprocessing.strip_non_alphanum)

        if 'stop' in self._prep_opts:
            # Only english language supported yet, gensim module
            current_app.logger.info('Remove stopwords...')
            clean_data = clean_data.apply(gensim.parsing.preprocessing.remove_stopwords)

        if 'stem' in self._prep_opts:
            # PorterStemmer used from gensim module
            current_app.logger.info('Stemming...')
            clean_data = clean_data.apply(gensim.parsing.stem_text)

        if 'lemma' in self._prep_opts:
            # WordNet Lemmatizer used
            current_app.logger.info('Apply WordNet Lemmatizer...')
            wn_lemma = WordNetLemmatizer()
            clean_data = clean_data.apply(lambda s: ' '.join([wn_lemma.lemmatize(word) for word in s.split()]))

        # Next result: Preprocessed data
        if train:
            self.model_feats.append(clean_data.head().values.tolist())

        # Store clean data
        return clean_data

    def _fit(self, data, opts):
        """
        Internal training method.
        """
        if self.algorithm != 'w2v':
            # TF-IDF
            current_app.logger.info('TF-IDF model')
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
                current_app.logger.info('KMeans clustering')
                corpus_tfidf = gensim.matutils.corpus2csc(tfidf_matrix).transpose()
                self.model = MiniBatchKMeans(n_clusters=opts.n_clusters.data, init='k-means++',
                                             max_iter=1000, n_init=1)
                self.model.fit(corpus_tfidf)
                self.model_feats.append(self.model)

            elif self.algorithm == 'lsi':
                current_app.logger.info('LSI model')
                self.model = gensim.models.lsimodel.LsiModel(corpus=tfidf_matrix, id2word=self.dictionary,
                                                             num_topics=opts.num_topics.data)
                self.model_feats.append(self.model.print_topics(opts.num_topics.data))

            elif self.algorithm == 'lda':
                current_app.logger.info('LDA model')
                self.model = gensim.models.ldamodel.LdaModel(corpus=tfidf_matrix, id2word=self.dictionary,
                                                             num_topics=opts.num_topics.data, update_every=1,
                                                             chunksize=10000, passes=1)
                self.model_feats.append(self.model.print_topics(opts.num_topics.data))

        elif self.algorithm == 'w2v':
            current_app.logger.info('Word2Vec model')
            self.model = gensim.models.Word2Vec(sentences=TokenizedSentences(data), size=opts.size.data,
                                                window=opts.window.data, min_count=opts.min_df.data, workers=4,
                                                max_vocab_size=opts.max_vocab.data, hs=1, negative=0)
            self.model_feats.append(self.model)
