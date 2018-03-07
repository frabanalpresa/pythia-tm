Pythia-TM
=========

## What is it?
Pythia-TM is a tool that contains several algorithms to be applied in textual data. With a user-friendly GUI
  and the power of open source tools in its backend, this tool aims to provide powerful capabilities for great
  document analysis, without a deep user knowledge in the field.

It is currently under development, and it is not yet heavily tested. Therefore, it is recommended to be used only for teaching purposes, and not for business-ready solutions, for which is definitely not advised.

## Main features
A list of main features would include:
- It is just made from open source software.
- It includes graphical tools to train, explore and score an unsupervised model on texts.
- Models can be easily saved, as well as scores, and be used in other codes.
- Explanations about the models and workflows implemented are provided in-app.

## Where to get it?
You can find the code for this app in the following URL:

[https://github.com/frabanalpresa/pythia-tm](https://github.com/frabanalpresa/pythia-tm)


## How to install it

It is recommended to use a virtual environment to test this tool. In order to do so, please make use of Conda
environments, or *virtualenv* tool. Making use of conda environments, a ```.yml``` file is provided to configure the environment:

```
git clone https://github.com/frabanalpresa/pythia-tm
cd textminer
conda env create -f environment.yml
source activate textminer
```

wherease using *virtualenv* tool is also possible for those who do not have a Conda distribution installed:

```
git clone https://github.com/frabanalpresa/pythia-tm
cd textminer
pip3 install virtualenv
virtualenv --python=python3 .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Dependencies
Tool was tested for Python 3.6 and the following module versions:

```
numpy==1.12.1
Flask_Session==0.3.1
nltk==3.2.4
bokeh==0.12.13
gensim==3.1.0
pandas==0.22.0
Flask_WTF==0.14.2
WTForms==2.1
Flask==0.12.2
scikit_learn==0.19.1
```

## Execute it
Once dependencies are met, just execute the main file in this repo:

```
python3 app.py
```

You can access the main page at [http://127.0.0.1:5000], and then navigate through the tool.

## License
This software is licensed under the terms of GNU LGPLv3. See the [LICENSE](LICENSE.txt) file for license rights and limitations.

## Why Pythia-TM?

[Pythia](https://en.wikipedia.org/wiki/Pythia) (Πῡθίᾱ in Ancient Greek) was the High Priestess of the Temple of Apollo in Delphi, also known as Oracle of Delphi, widely credited for her prophecies. Its name is also related to Python as she was considered the House of the Snakes. It diverges from the origin of Python programming language name, which comes from Monty Python comedy group.

The suffix TM comes from Text Mining acronym, the field covered by this piece of software.
