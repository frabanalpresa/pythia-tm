from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import FloatField, IntegerField, SelectMultipleField, StringField
from wtforms.validators import Optional, required


class SearchForm(FlaskForm):
    input_file = FileField(validators=[FileRequired(), FileAllowed(['csv'], 'CSV files only!')], label='Data filename')
    model_file = StringField(validators=[Optional()], label='Model filename (*.pythia) - Leave blank if not saved')
    prep_opts = SelectMultipleField('Preprocessing options',
                                    choices=[('clean', 'Text Cleaning'), ('stop', 'Remove Stopwords'),
                                             ('stem', 'Stemming'), ('lemma', 'Lemmatize')],
                                    option_widget=[Optional()])


class TfIdfForm(SearchForm):
    min_df = IntegerField('Min count', [required()])
    max_df = FloatField('Max ratio', [required()])
    max_feats = IntegerField('Max feats', [required()])


class KmeansForm(TfIdfForm):
    n_clusters = IntegerField('K', [required()])


class LsiForm(TfIdfForm):
    num_topics = IntegerField('NTopics', [required()])


class W2vForm(SearchForm):
    min_df = IntegerField('Min count', [required()])
    size = IntegerField('Size', [required()])
    window = IntegerField('Window', [required()])
    max_vocab = IntegerField('Max vocab size', [required()])


class W2vExploreForm(FlaskForm):
    pos_words = StringField(validators=[Optional()], label='Positive words (e.g. tea green)')
    neg_words = StringField(validators=[Optional()], label='Negative words (e.g. red)')


class ExploreMainForm(FlaskForm):
    model_file = FileField(validators=[FileRequired(), FileAllowed(['pythia'], '.pythia files only!')],
                           label='Model filename (default ./models/*.pythia)')

class ScoreMainForm(FlaskForm):
    input_file = FileField(validators=[FileRequired(), FileAllowed(['csv'], 'CSV files only!')], label='Data filename')
    model_file = FileField(validators=[FileRequired(), FileAllowed(['pythia'], '.pythia files only!')],
                           label='Model filename (default ./models/*.pythia)')

class ScoreSaveForm(FlaskForm):
    input_file = StringField(validators=[Optional()], label='Scores full path (*.csv)')
