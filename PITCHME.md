@title[Main slide]

## Pythia-TM

<span style="font-size:0.6em; color:gray">What is it?</span> |
<span style="font-size:0.6em; color:gray">Why even use it?</span>

---

[https://gitpitch.com/frabanalpresa/pythia-tm/master](https://gitpitch.com/frabanalpresa/pythia-tm/master)

<br>

[https://github.com/frabanalpresa/pythia-tm](https://github.com/frabanalpresa/pythia-tm)

---
@title[YATM tool]

### Yet Another Text Mining (YATM) tool

There are plenty of solutions to perform TM tasks in the market. This one is focused on explaining each part inside
the app, and easily generating results, providing a good start for iterating towards the desired solution.

@fa[arrow-down]
+++

While focusing on classic NLP algorithms, it is good to notice that some cutting-edge algorithms can also be included,
and it is a good way to incorporate state-of-the-art knowledge into graphical tools.


@fa[arrow-down]
+++
@title[Main features]

**Main features**

<table style="color:gray; font-size:1em">
  <tr>
    <td>Made of Open Source software</td>
  </tr>
  <tr class="fragment">
    <td>GUI interface to facilitate analysis</td>
  </tr>
  <tr class="fragment">
    <td>Train, explore, score (reuse)</td>
  </tr>
  <tr class="fragment">
    <td>In-app explanations</td>
  </tr>
</table>


@fa[arrow-down]
+++

**License**

- [GNU LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.en.html)

<br>

- [Why?](https://choosealicense.com/licenses/lgpl-3.0/)

<table style="color:gray; font-size:0.8em">
  <tr>
    <td>**Uses**</td>
    <td>Commercial</td>
    <td>Patent</td>
    <td>Private</td>
  </tr>
  <tr class="fragment">
    <td>**Limit**</td>
    <td>Warranty</td>
    <td>Liability</td>
    <td></td>
  </tr>
</table>

<span style="font-size:0.6em; color:gray">Disclose source</span> |
<span style="font-size:0.6em; color:gray">License modifications (same)</span>

---
@title[Train]

### Train a model

- In the main page, press "Go" in "Train your model".
- Select workflow to apply
- Fill the form with the required information.
- Press "Go!"

@fa[arrow-down]
+++
@title[Models implemented]

**Current models implemented**

<table style="color:gray; font-size:0.8em">
  <tr>
    <th>Preprocess</th>
    <th>Modeling</th>
    <th>Algorithms</th>
  </tr>
  <tr class="fragment">
    <td>Text cleaning</td>
    <td>BoW + TF-IDF</td>
    <td>K-Means</td>
  </tr>
  <tr class="fragment">
    <td>Stopwords</td>
    <td>Word2Vec</td>
    <td>LSI</td>
  </tr>
  <tr class="fragment">
    <td>Lemmatization</td>
    <td></td>
    <td>LDA</td>
  </tr>
  <tr class="fragment">
    <td>Stemming</td>
    <td></td>
    <td></td>
  </tr>
</table>


@fa[arrow-down]
+++

**Save your model**

In the form to be filled out before training the model, you can input an absolute path to wherever you want the model saved. It is not required, but it is good to have models saved for iterating fast.

---
@title[Explore]

### Explore your model

- In the main page, press "Go" in "Explore a model".
- Choose where model is found in your local filesystem.
- Press Go!

@fa[arrow-down]
+++
@title[What to expect]

**What to expect?**

<table style="color:gray; font-size:1em">
  <tr>
    <td>Sample of original data</td>
  </tr>
  <tr class="fragment">
    <td>Preprocessed samples</td>
  </tr>
  <tr class="fragment">
    <td>Clustering components</td>
  </tr>
  <tr class="fragment">
    <td>Topic components</td>
  </tr>
  <tr class="fragment">
    <td>Word similarities</td>
  </tr>
</table>

---
@title[Score]

### Score over new data

- In the main page, press "Go" in "Score new data".
- Choose where data and model are found in your local filesystem.
- Press Go!

@fa[arrow-down]
+++
@title[Explore the scores]

**Explore your scores**

<table style="color:gray; font-size:1em">
  <tr>
    <td>Score distribution</td>
  </tr>
  <tr class="fragment">
    <td>Clustering main components</td>
  </tr>
  <tr class="fragment">
    <td>Topic components</td>
  </tr>
</table>

@fa[arrow-down]
+++
@title[Save the scores]

**Save the scores**

If you need your scores later on, you can save them as a *CSV* file. In this file, you will find the score for each row provided as input, in the same order.

---
@title[Final comments]

### Comments

<br>

<table style="color:gray; font-size:1em">
  <tr>
    <td>Work under development</td>
  </tr>
  <tr class="fragment">
    <td>Many restrictions yet</td>
  </tr>
  <tr class="fragment">
    <td>Mainly illustrative purposes</td>
  </tr>
</table>

---
@title[A final word]

### A final word

It may not be the best coded piece of software, definitely not the best documented, nor the best fitted for usual business problems... but you can make your contribution to it!

<hr>

Open an issue, or much better, fork the repository and place a merge request when you finish developing a new feature!
