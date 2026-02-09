from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

MOVIE_METADATA_PATH = "data/movie_titles_metadata.txt"

def plot_movies_per_genre():
    genre_counter = Counter()
    with open(MOVIE_METADATA_PATH, encoding="iso-8859-1") as file:
        for line in file:
            parsed_line = line.split("+++$+++")
            genres = eval(parsed_line[-1].strip())
            for g in genres:
                genre_counter[g] += 1
    sorted_genre_counts = genre_counter.most_common()
    sorted_genre_counts.reverse()
    bars = plt.barh([x[0] for x in sorted_genre_counts], [x[1] for x in sorted_genre_counts])
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")
    plt.bar_label(bars)
    plt.show()

def plot_genres_per_movie():
    with open(MOVIE_METADATA_PATH, encoding="iso-8859-1") as file:
        genre_lists = [eval(line.split("+++$+++")[-1].strip()) for line in file]
    len_genres = [len(gl) for gl in genre_lists]
    _, _, bars = plt.hist(len_genres, bins=np.arange(0,12), edgecolor="black")
    plt.xlabel("Number of Genres")
    plt.xticks(range(0,12))
    plt.ylabel("Frequency")
    plt.bar_label(bars)
    plt.show()
    print(sum(len_genres) / len(len_genres))

# Based on code found here: https://discuss.pytorch.org/t/how-to-create-a-multilabel-confusion-matrix-for-14-disease-classes-in-pytorch/211491
def plot_error_matrix(all_labels, all_predictions, classes, normalize=True):
    num_samples, num_classes = np.shape(all_labels)
    matrix = np.zeros((num_classes, num_classes), dtype = float if normalize else int)
    true_totals = np.zeros(num_classes)

    for i in range(num_samples):
        true_labels = np.where(all_labels[i] == 1)[0]
        pred_labels = np.where(all_predictions[i] == 1)[0]
        for t in true_labels:
            true_totals[t] += 1
            for p in pred_labels:
                if t == p or t not in pred_labels or p not in true_labels:
                    matrix[t, p] += 1

    if normalize:
        for t in range(num_classes):
            for p in range(num_classes):
                matrix[t, p] /= true_totals[t]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        xlabel='Predicted Label',
        ylabel='True Label',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, format(matrix[i, j], fmt),
                ha='center', va='center',
                color='white' if matrix[i, j] > thresh else 'black'
            )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_movies_per_genre()
