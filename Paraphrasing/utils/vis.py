import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import numpy as np

def plot_accuracy_bar(df, language='yoruba', save_path=None):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Train Dataset', y='Accuracy', hue='Test Dataset', data=df)
    plt.title(f'Model Generalization Performance - {language}')
    plt.xlabel("Train Dataset")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_embedding_2D(embeddings, labels, title, method='tsne', save_path=None):
    reducer = TSNE(n_components=2) if method == 'tsne' else umap.UMAP(n_components=2)
    reduced = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idxs = np.where(labels == label)
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=f"Class {label}", alpha=0.6)
    plt.legend()
    plt.title(f"{method.upper()} Projection of {title}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
