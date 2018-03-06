import io
import pickle
from flask import Blueprint, render_template, session, redirect, url_for
from forms import ExploreMainForm, W2vExploreForm

explore_page = Blueprint('explore_page', __name__, template_folder='templates')


@explore_page.route('/explore', methods=('GET', 'POST'))
def explore_main():
    form = ExploreMainForm()
    if form.validate_on_submit():
        model = pickle.load(io.BytesIO(form.model_file.data.read()))
        session['model_feats'] = model.explore()
        if model.algorithm == 'kmeans':
            return redirect(url_for('explore_page.explore_kmeans'))
        elif model.algorithm == 'lsi':
            return redirect(url_for('explore_page.explore_lsi'))
        elif model.algorithm == 'lda':
            return redirect(url_for('explore_page.explore_lda'))
        else:
            return redirect(url_for('explore_page.explore_w2v'))

    return render_template('items/explore_main.html', form=form)


@explore_page.route('/explore/kmeans', methods=('GET', 'POST'))
def explore_kmeans():
    res = session['model_feats']
    model_centroids = res[3].cluster_centers_.argsort()[:, ::-1]
    centroids = dict()
    for n in range(20):
        centroids[n] = list()
        for ind in model_centroids[n, :10]:
            centroids[n].append(list(res[2].token2id.keys())[ind])
        centroids[n] = ', '.join(centroids[n])

    return render_template('items/explore_model.html', orig_data=res[0], clean_data=res[1],
                           dict_results=res[2].token2id, model_results=centroids, algorithm='kmeans')


@explore_page.route('/explore/lsi', methods=('GET', 'POST'))
def explore_lsi():
    res = session['model_feats']

    topics = dict()
    for i_topic in res[3]:
        topics[i_topic[0]] = i_topic[1]
    return render_template('items/explore_model.html', orig_data=res[0], clean_data=res[1],
                           dict_results=res[2].token2id, model_results=topics, algorithm='lsi')


@explore_page.route('/explore/lda', methods=('GET', 'POST'))
def explore_lda():
    res = session['model_feats']

    topics = dict()
    for i_topic in res[3]:
        topics[i_topic[0]] = i_topic[1]
    return render_template('items/explore_model.html', orig_data=res[0], clean_data=res[1],
                           dict_results=res[2].token2id, model_results=topics, algorithm='lda')


@explore_page.route('/explore/w2v', methods=('GET', 'POST'))
def explore_w2v():
    res = session['model_feats']
    model_results = None
    form = W2vExploreForm()
    if form.validate_on_submit():
        model_results = res[-1].wv.most_similar(positive=form.pos_words.data.split(),
                                                negative=form.neg_words.data.split())
        model_results = [(i_res[0], round(i_res[1], 6)) for i_res in model_results]

    return render_template('items/explore_model.html', orig_data=res[0], clean_data=res[1],
                           dict_results=None, model_results=model_results, algorithm='w2v', form=form)
