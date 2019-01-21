"""
Module: explore page.

Compiles all functions to explore the results
obtained after a pythia-tm model is trained.

A new function is defined per algorithm, as
different algorithms may require different
metrics or KPIs to be explored.
"""
import io
import pickle
from flask import Blueprint, render_template, session, redirect, url_for
from forms import ExploreMainForm, W2vExploreForm

# Define blueprint
explore_page = Blueprint('explore_page', __name__, template_folder='templates')

# Main method
@explore_page.route('/explore', methods=('GET', 'POST'))
def explore_main():
    """
    Main function for exploring models web app. It redirects
    to the specific model explore function regarding the
    model algorithm.
    """
    form = ExploreMainForm()
    if form.validate_on_submit():
        model = pickle.load(io.BytesIO(form.model_file.data.read()))
        session['model_feats'] = model.explore()
        return redirect(url_for(f'explore_page.explore_{model.algorithm}'))
    return render_template('items/explore_main.html', form=form)


@explore_page.route('/explore/kmeans', methods=('GET', 'POST'))
def explore_kmeans():
    """
    KMeans explore function.

    It explores the most important terms in each cluster, up
    to 10 per cluster, up to 20 clusters.
    """
    res = session['model_feats']
    model_centroids = res[3].cluster_centers_.argsort()[:, ::-1]
    centroids = dict()
    for i_cluster in range(min(20, model_centroids.shape[0])):
        centroids[i_cluster] = list()
        for ind in model_centroids[i_cluster, :min(10, model_centroids.shape[1])]:
            centroids[i_cluster].append(list(res[2].token2id.keys())[ind])
        centroids[i_cluster] = ', '.join(centroids[i_cluster])

    return render_template('items/explore_model.html', orig_data=res[0], clean_data=res[1],
                           dict_results=res[2].token2id, model_results=centroids, algorithm='kmeans')


@explore_page.route('/explore/lsi', methods=('GET', 'POST'))
def explore_lsi(algorithm='lsi'):
    """
    LSI explore function.

    Explores the topics composition, as well as the weight of each
    term composing the topic.
    """
    res = session['model_feats']

    topics = dict()
    for i_topic in res[3]:
        topics[i_topic[0]] = i_topic[1]
    return render_template('items/explore_model.html', orig_data=res[0], clean_data=res[1],
                           dict_results=res[2].token2id, model_results=topics, algorithm=algorithm)


@explore_page.route('/explore/lda', methods=('GET', 'POST'))
def explore_lda():
    """
    LDA explore function.

    Explores the topics composition, as well as the weight of each
    term composing the topic.
    """
    return explore_lsi('lda')


@explore_page.route('/explore/w2v', methods=('GET', 'POST'))
def explore_w2v():
    """
    Word2Vec explore function.

    Allows to explore the words related to others in a positive
    and negative ways. Positive and negative words have to be
    in the dictionary for Word2Vec.
    """
    res = session['model_feats']
    model_results = None
    form = W2vExploreForm()
    if form.validate_on_submit():
        model_results = res[-1].wv.most_similar(positive=form.pos_words.data.split(),
                                                negative=form.neg_words.data.split())
        model_results = [(i_res[0], round(i_res[1], 6)) for i_res in model_results]

    return render_template('items/explore_model.html', orig_data=res[0], clean_data=res[1],
                           dict_results=None, model_results=model_results, algorithm='w2v', form=form, 
                           vocab=list(session['model'].model.wv.vocab.keys()))
