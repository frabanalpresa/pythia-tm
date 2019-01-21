"""
Module: scoring page.

Compiles all functions to scores models trained
with pythia-tm library.

A new function is defined per algorithm, as
different algorithms may require different
parameters, and scoring visual results
may be different in different algorithms.
"""
import os
import io
import pickle
import flask
import numpy as np
from bokeh.embed import components
from bokeh.plotting import figure
from flask import Blueprint, render_template, session, redirect, url_for, request, flash
from forms import ScoreMainForm, ScoreSaveForm

# Define blueprint
score_page = Blueprint('score_page', __name__, template_folder='templates')


def plot_dist(data, bins=None):
    """Generate score distribution plot to be displayed in the html page."""
    bins = 50 if bins is None else bins
    hist, edges = np.histogram(data[data != 0], density=True, bins=bins)

    # overall size and look of graph
    plt = figure(title='Score ditribution', plot_width=1024, plot_height=480)
    plt.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#036564", line_color="#033649")

    # render template
    script, div = components(plt)
    return script, div


@score_page.route('/score', methods=('GET', 'POST'))
def score_main():
    """Main function to score a test set, given a trained model."""
    form = ScoreMainForm()
    if form.validate_on_submit():
        model = pickle.load(io.BytesIO(form.model_file.data.read()))
        scores = model.predict(form)
        session['model'] = model
        session['scores'] = scores
        if model.algorithm in ('kmeans', 'lsi', 'lda', 'w2v'):
            return redirect(url_for(f'score_page.score_{model.algorithm}'))

    return render_template('items/score_main.html', form=form)


@score_page.route('/score/save', methods=('GET', 'POST'))
def save_scores():
    """Function to save scoring results of a test set."""
    form = ScoreSaveForm()
    if flask.request.method == 'GET':
        # Initialize model name if page is loaded,
        # but use the one from user if form is submitted.
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
    """
    KMeans scoring function.

    Apart from scoring, show scores distribution and cluster members
    (up to 20 clusters, up to 10 terms per cluster).
    """
    res = session['model']
    model_centroids = res.model.cluster_centers_.argsort()[:, ::-1]
    centroids = dict()
    for i_cluster in range(min(20, model_centroids.shape[0])):
        centroids[i_cluster] = list()
        for ind in model_centroids[i_cluster, :min(10, model_centroids.shape[1])]:
            centroids[i_cluster].append(list(res.dictionary.token2id.keys())[ind])
        centroids[i_cluster] = ', '.join(centroids[i_cluster])

    script, div = plot_dist(session['scores'], session['model'].model.n_clusters)
    return render_template('items/score_clustering.html', algorithm='kmeans', centroids=centroids,
                           script=script, div=div)


@score_page.route('/score/lsi', methods=('GET', 'POST'))
def score_lsi():
    """
    LSI scoring function.

    Show generic information on scores, or information on given topic
    selected by user.
    """
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
    """
    LDA scoring function.

    Show generic information on scores, or information on given topic
    selected by user.
    """
    return score_lsi()


@score_page.route('/score/w2v', methods=('GET', 'POST'))
def score_w2v():
    """
    W2V scoring function.

    Show generic information on score distribution.
    """
    script, div = plot_dist(session['scores'])
    return render_template('items/score_w2v.html', algorithm='w2v', script=script, div=div)
