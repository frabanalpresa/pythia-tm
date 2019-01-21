"""
Pythia-TM text mining module.

Web application for education purposes that
illustrates text mining complete workflows
through a graphical interface.

Author is not responsible for other applications
of this module, use at your own risk.

Author: Fernando Rabanal
Version: 0.2
Date: 2019-01-17
License: LGPLv3
"""
from flask import Flask, render_template
from flask_session import Session
from train_app import train_page
from explore_app import explore_page
from score_app import score_page

# Flask
app = Flask(__name__)

# Blueprints
app.register_blueprint(train_page)
app.register_blueprint(explore_page)
app.register_blueprint(score_page)

# Flask Session: check Configuration section for more details
app.config.from_object(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'sessions'
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = '7dbb-cee1-80ba-4d45-6e4a-a1cf-bdad-9c7b'
Session(app)

@app.route('/', methods=('GET',))
def main_page():
    """Main page"""
    return render_template('items/main.html')

@app.route('/about', methods=('GET',))
def about_page():
    """About me page"""
    return render_template('items/about.html')


if __name__ == '__main__':
    app.run()
