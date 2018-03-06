import os
import io
import pickle
import numpy as np
from flask import Blueprint, render_template, session, redirect, url_for, request, flash
from forms import ScoreMainForm, ScoreSaveForm

from bokeh.embed import components
from bokeh.plotting import figure

score_page = Blueprint('score_page', __name__, template_folder='templates')


def plot_dist(data, bins=None):
    if bins is None:
        hist, edges = np.histogram(data[data != 0], density=True, bins=50)
    else:
        hist, edges = np.histogram(data[data != 0], density=True, bins=bins)

    # overall size and look of graph
    p = figure(title='Score ditribution', plot_width=1024, plot_height=480)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#036564", line_color="#033649")

    # render template
    script, div = components(p)
    return script, div


@score_page.route('/score', methods=('GET', 'POST'))
def score_main():
    form = ScoreMainForm()
    if form.validate_on_submit():
        model = pickle.load(io.BytesIO(form.model_file.data.read()))
        scores = model.predict(form)
        session['model'] = model
        session['scores'] = scores
        if model.algorithm == 'kmeans':
            return redirect(url_for('score_page.score_kmeans'))
        elif model.algorithm == 'lsi':
            return redirect(url_for('score_page.score_lsi'))
        elif model.algorithm == 'lda':
            return redirect(url_for('score_page.score_lda'))
        else:
            return redirect(url_for('score_page.score_w2v'))

    return render_template('items/score_main.html', form=form)


@score_page.route('/score/save', methods=('GET', 'POST'))
def save_scores():
    form = ScoreSaveForm()
    form.input_file.data = f'{os.getcwd()}/results/scores.csv'
    if form.validate_on_submit():
        if form.input_file.data.startswith('~'):
            form.input_file.data = os.path.expanduser(form.input_file.data)
        np.savetxt(form.input_file.data, session['scores'], fmt='%.6f', delimiter=',')
        flash('Scores saved successfully!')
        return redirect(url_for('main_page'))

    return render_template('items/score_save.html', form=form)


@score_page.route('/score/kmeans', methods=('GET', 'POST'))
def score_kmeans():
    res = session['model']
    model_centroids = res.model.cluster_centers_.argsort()[:, ::-1]
    centroids = dict()
    for n in range(20):
        centroids[n] = list()
        for ind in model_centroids[n, :10]:
            centroids[n].append(list(res.dictionary.token2id.keys())[ind])
        centroids[n] = ', '.join(centroids[n])

    script, div = plot_dist(session['scores'], session['model'].model.n_clusters)
    return render_template('items/score_clustering.html', algorithm='kmeans', centroids=centroids,
                           script=script, div=div)


@score_page.route('/score/lsi', methods=('GET', 'POST'))
def score_lsi():
    pos_topics = range(session['model'].model.num_topics)
    if request.method == 'POST':
        topic_info = sorted(session['model'].model.show_topic(int(request.form['sel_topics'])),
                            key=lambda x: x[1])[::-1]
        script, div = plot_dist(session['scores'][:, int(request.form['sel_topics'])])

        return render_template('items/score_topics.html', algorithm='lsi', drop_topics=pos_topics,
                               sel_topic=int(request.form['sel_topics']), topic_info=topic_info, script=script,
                               div=div)

    return render_template('items/score_topics.html', algorithm='lsi', drop_topics=pos_topics, sel_topic=None,
                           topic_info=None, script=None, div=None)


@score_page.route('/score/lda', methods=('GET', 'POST'))
def score_lda():
    pos_topics = range(session['model'].model.num_topics)
    if request.method == 'POST':
        topic_info = sorted(session['model'].model.show_topic(int(request.form['sel_topics'])),
                            key=lambda x: x[1])[::-1]
        script, div = plot_dist(session['scores'][:, int(request.form['sel_topics'])])

        return render_template('items/score_topics.html', algorithm='lsi', drop_topics=pos_topics,
                               sel_topic=int(request.form['sel_topics']), topic_info=topic_info, script=script,
                               div=div)

    return render_template('items/score_topics.html', algorithm='lsi', drop_topics=pos_topics, sel_topic=None,
                           topic_info=None, script=None, div=None)


@score_page.route('/score/w2v', methods=('GET', 'POST'))
def score_w2v():
    script, div = plot_dist(session['scores'])
    return render_template('items/score_w2v.html', algorithm='w2v', script=script, div=div)
