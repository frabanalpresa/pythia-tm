import os
import pickle
from flask import Blueprint, render_template, session, redirect, url_for
from forms import *
from pythia.pythia_tm import PythiaTM

train_page = Blueprint('train_page', __name__, template_folder='templates')


@train_page.route('/train')
def train_main():
    return render_template('items/train_main.html')


def train_fun(algorithm, form):
    model = PythiaTM(algorithm)
    model.train(form)

    # Save
    if form.model_file.data.endswith('.pythia'):
        filepath = f'{form.model_file.data}'
        pickle.dump(model, open(filepath, 'wb'))

    return model.explore()


@train_page.route('/train/kmeans', methods=('GET', 'POST'))
def train_kmeans():
    form = KmeansForm()
    form.model_file.data = f'{os.getcwd()}/models/mymodel.pythia'
    if form.validate_on_submit():
        session['model_feats'] = train_fun('kmeans', form)
        return redirect(url_for('explore_page.explore_kmeans'))
    return render_template('items/train_model.html', alg='kmeans', form=form)


@train_page.route('/train/lsi', methods=('GET', 'POST'))
def train_lsi():
    form = LsiForm()
    form.model_file.data = f'{os.getcwd()}/models/mymodel.pythia'
    if form.validate_on_submit():
        session['model_feats'] = train_fun('lsi', form)
        return redirect(url_for('explore_page.explore_lsi'))
    return render_template('items/train_model.html', alg='lsi', form=form)


@train_page.route('/train/lda', methods=('GET', 'POST'))
def train_lda():
    form = LsiForm()
    form.model_file.data = f'{os.getcwd()}/models/mymodel.pythia'
    if form.validate_on_submit():
        session['model_feats'] = train_fun('lda', form)
        return redirect(url_for('explore_page.explore_lda'))
    return render_template('items/train_model.html', alg='lda', form=form)


@train_page.route('/train/w2v', methods=('GET', 'POST'))
def train_w2v():
    form = W2vForm()
    form.model_file.data = f'{os.getcwd()}/models/mymodel.pythia'
    if form.validate_on_submit():
        session['model_feats'] = train_fun('w2v', form)
        return redirect(url_for('explore_page.explore_w2v'))
    return render_template('items/train_model.html', alg='w2v', form=form)
