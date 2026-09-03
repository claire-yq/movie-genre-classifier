# Movie Genre Classifier

Predicting a movie's genre(s) from its dialogue, using data from the [Cornell Movie-Dialogs Corpus](https://convokit.cornell.edu/documentation/movie.html).

## Repository Contents

| File | Description |
|---|---|
| `main.ipynb` | Main notebook that loads the data, runs preprocessing, trains models, and reports results |
| `process.py` | Data preprocessing: builds character documents from dialogue, filters genres and stopwords, and converts text into unigram or average-embedding features |
| `train_eval.py` | Model training and evaluation, including a dev-set grid search over Naive Bayes and Logistic Regression configurations and final test-set scoring |
| `visualize.py` | Plotting utilities — genre distribution across movies, number of genres per movie, and a per-class error/confusion matrix |
| `paper.pdf` | Write-up describing the approach, experiments, and results |
