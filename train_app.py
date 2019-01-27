"""
Module: train page.

Compiles all functions to train models with
pythia-tm library.

A new function is defined per algorithm, as
different algorithms may require different
parameters.
"""
import os
import pickle
import flask
from flask import Blueprint, current_app, render_template, session, redirect, url_for
from forms import KmeansForm, LsiForm, W2vForm
from pythia.pythia_tm import PythiaTM

# Define blueprint
train_page = Blueprint('train_page', __name__, template_folder='templates')


@train_page.route('/train')
def train_main():
    """Main training page"""
    return render_template('items/train_main.html')


def train_fun(algorithm, form):
    """
    Training function.

    Wrapper over PythiaTM (pythia.pythia_tm module) library.
    It also saves the model and returns the necessary
    information to explore it.
    """
    # Instantiate a function, and train the model.
    model = PythiaTM(algorithm)
    model.train(form)
    session['model'] = model

    # Save the model in the provided path
    if form.model_file.data.endswith('.pythia'):
        filepath = f'{form.model_file.data}'
        pickle.dump(model, open(filepath, 'wb'))

    # Explore the model at the end.
    return model.explore()

@train_page.route('/train/model', methods=('GET', 'POST'))
def train_model():
    algorithm = flask.request.args.get('arg')
    if algorithm == 'kmeans':
        form = KmeansForm()
    elif algorithm in ['lsi', 'lda']:
        form = LsiForm()
    elif algorithm == 'w2v':
        form = W2vForm()

    if flask.request.method == 'GET':
        # Initialize model name if page is loaded,
        # but use the one from user if form is submitted.
        form.model_file.data = f'{os.getcwd()}/models/mymodel.pythia'

    if form.validate_on_submit():
        session['model_feats'] = train_fun(algorithm, form)
        return redirect(url_for(f'explore_page.explore_{algorithm}'))
    return render_template('items/train_model.html', alg=algorithm, form=form)
