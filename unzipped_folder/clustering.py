# clustering.py
# === clustering.py ===
import numpy as np
import os
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

OUTPUT_DIR = os.getcwd()
print("Saving plots to:", OUTPUT_DIR)

def evaluate(features, labels):
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return None
    feats = features[mask]
    lbls = labels[mask]
    return {
        'silhouette': silhouette_score(feats, lbls),
        'calinski_harabasz': calinski_harabasz_score(feats, lbls),
        'davies_bouldin': davies_bouldin_score(feats, lbls),
        'n_clusters': len(set(lbls)),
        'n_noise': int((labels == -1).sum())
    }

def apply_clustering(df, methods=None, dbscan_params=None, hdbscan_params=None, optimize=False):
    if methods is None:
        methods = ['dbscan', 'hdbscan']

    X = df[['degree','entropy']].copy()
    X['degree'] = np.log1p(X['degree'])
    features = StandardScaler().fit_transform(X)

    print("Feature matrix shape:", features.shape)

    if hdbscan_params is None:
        hdbscan_params = {
            'min_cluster_size': max(5, int(len(features) * 0.1)),
            'min_samples': 2
        }

    if dbscan_params is None:
        dbscan_params = {
            'eps': 1.0,
            'min_samples': 5
        }

    if 'dbscan' in methods:
        db = DBSCAN(**dbscan_params)
        lbl = db.fit_predict(features)
        df['dbscan_cluster'] = lbl
        print("DBSCAN evaluation:", evaluate(features, lbl))

        plt.figure(figsize=(10, 6))
        plt.scatter(features[:, 0], features[:, 1], c=lbl, cmap='tab10', alpha=0.6)
        plt.title("DBSCAN Clustering")
        plt.xlabel("Log Degree")
        plt.ylabel("Entropy")
        plt.grid(True)
        output_path = os.path.join(OUTPUT_DIR, "dbscan_clusters.png")
        print(f"Saving DBSCAN plot to {output_path}")
        plt.savefig(output_path, dpi=300)
        plt.close()

    if 'hdbscan' in methods:
        hdb = hdbscan.HDBSCAN(**hdbscan_params)
        lbl = hdb.fit_predict(features)
        df['hdbscan_cluster'] = lbl
        print("HDBSCAN evaluation:", evaluate(features, lbl))

        plt.figure(figsize=(10, 6))
        plt.scatter(features[:, 0], features[:, 1], c=lbl, cmap='tab10', alpha=0.6)
        plt.title("HDBSCAN Clustering")
        plt.xlabel("Log Degree")
        plt.ylabel("Entropy")
        plt.grid(True)
        output_path = os.path.join(OUTPUT_DIR, "hdbscan_clusters.png")
        print(f"Saving HDBSCAN plot to {output_path}")
        plt.savefig(output_path, dpi=300)
        plt.close()

    return df
