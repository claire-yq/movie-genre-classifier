import re
from collections import defaultdict, Counter
import ast
from nltk.corpus import stopwords
import numpy as np

DISCARDED_GENRES = ["adult", "documentary", "film-noir", "short"] # remove because we don't have enough data

def get_char_names(characters):
    names = defaultdict(list)
    for char in characters:
        char_names = char.retrieve_meta("character_name").lower().split() # if name is more than one word, add each word separately
        names[char.retrieve_meta("movie_idx")].extend(char_names)
    return names

def preprocess(characters):
    X = []
    y = []
    char_names_by_movie = get_char_names(characters)  # maps movie idx to character names
    sw = stopwords.words("english")

    for char in characters:
        genres = next(char.iter_conversations()).retrieve_meta("genre")
        processed_genres = list(filter(lambda g: g not in DISCARDED_GENRES, ast.literal_eval(genres)))
        if len(processed_genres) == 0:
            continue

        movie_idx = char.retrieve_meta("movie_idx")
        sentences = [utt.text for utt in char.iter_utterances()]
        words = [w.lower() for s in sentences for w in re.findall(r'\w+', s)]
        filtered_words = list(filter(lambda w: w not in char_names_by_movie[movie_idx] and w not in sw, words))
        if len(filtered_words) > 0:
            X.append(filtered_words)
            y.append(processed_genres)

    return X,y

def get_unigrams(word_lists):
    return [Counter(wl) for wl in word_lists]

def get_average_embeddings(word_lists, embeddings, dim_count):
    avg_embs = []
    for wl in word_lists:
        word_vectors = [embeddings[w] for w in wl if w in embeddings]
        avg_embs.append(np.mean(np.array(word_vectors), axis=0) if len(word_vectors) > 0 else np.zeros(dim_count))
    return np.array(avg_embs)