"""
Module: form classes definition

Compiles all classes used in the different
parts of the web application to interact
with the user.
"""
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import FloatField, IntegerField, SelectMultipleField, StringField
from wtforms.validators import InputRequired, Optional, required


class SearchForm(FlaskForm):
    """Common search form for various algorithms"""
    input_file = FileField(validators=[FileRequired(), FileAllowed(['csv'], 'CSV files only!')], label='Data filename')
    text_field = StringField(validators=[InputRequired()], label='Text field')
    model_file = StringField(validators=[Optional()], label='Model filename (*.pythia) - Leave blank if not saved')
    prep_opts = SelectMultipleField('Preprocessing options',
                                    choices=[('clean', 'Text Cleaning'), ('stop', 'Remove Stopwords'),
                                             ('stem', 'Stemming'), ('lemma', 'Lemmatize')],
                                    option_widget=[Optional()])


class TfIdfForm(SearchForm):
    """Specific form fields for algorithms with TF-IDF."""
    min_df = IntegerField('Min count', [required()])
    max_df = FloatField('Max ratio', [required()])
    max_feats = IntegerField('Max feats', [required()])


class KmeansForm(TfIdfForm):
    """TF-IDF fields + KMeans specific one"""
    n_clusters = IntegerField('K', [required()])


class LsiForm(TfIdfForm):
    """TF-IDF fields + LSI specific one"""
    num_topics = IntegerField('NTopics', [required()])


class W2vForm(SearchForm):
    """Generic search fields + W2V specific ones"""
    min_df = IntegerField('Min count', [required()])
    size = IntegerField('Size', [required()])
    window = IntegerField('Window', [required()])
    max_vocab = IntegerField('Max vocab size', [required()])


class W2vExploreForm(FlaskForm):
    """Specific form to explore W2V models."""
    pos_words = StringField(validators=[Optional()], label='Positive words (e.g. tea green)')
    neg_words = StringField(validators=[Optional()], label='Negative words (e.g. red)')


class ExploreMainForm(FlaskForm):
    """Specific form to explore non-W2V models."""
    model_file = FileField(validators=[FileRequired(), FileAllowed(['pythia'], '.pythia files only!')],
                           label='Model filename (default ./models/*.pythia)')

class ScoreMainForm(FlaskForm):
    """Specific form to score models."""
    input_file = FileField(validators=[FileRequired(), FileAllowed(['csv'], 'CSV files only!')], label='Data filename')
    text_field = StringField(validators=[InputRequired()], label='Text field')
    model_file = FileField(validators=[FileRequired(), FileAllowed(['pythia'], '.pythia files only!')],
                           label='Model filename (default ./models/*.pythia)')

class ScoreSaveForm(FlaskForm):
    """Specific form to save model scores."""
    input_file = StringField(validators=[Optional()], label='Scores full path (*.csv)')
