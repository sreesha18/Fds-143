# %% [markdown]
# # Fundamentals of Data Science - Case Study 1: Online Retail Baskets
# **Name:** Sreesha CH | **Roll No:** 160124733143
#
# Workflow: inspect -> prepare -> explore -> PCA -> clustering -> evaluate -> interpret.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

RANDOM_STATE = 42
DATA_PATH = "Market_Basket_Optimisation.csv"   # change to your file name/path

# %% [markdown]
# ## 1. Load and inspect the data
# Each row is one transaction; products are spread over 20 columns and short
# baskets leave blank cells. header=0 treats the first row as column labels
# (this gives 7,500 rows). Use header=None if you want to keep the first row as a basket.

# %%
df = pd.read_csv(DATA_PATH, header=0)
print("Shape:", df.shape)
df.head()

# %%
total_cells = df.shape[0] * df.shape[1]
empty_cells = int(df.isna().sum().sum())
print("Transactions        :", df.shape[0])
print("Product slots / row :", df.shape[1])
print("Empty cells         :", empty_cells, f"({empty_cells / total_cells:.1%})")
print("Non-empty entries   :", total_cells - empty_cells)

# %% [markdown]
# ## 2. Data preparation
# Blank cells are dropped (no fake product name inserted), whitespace is trimmed,
# and each row becomes a list of products (a basket).

# %%
def row_to_basket(row):
    items = []
    for v in row:
        if pd.notna(v):
            s = str(v).strip()
            if s:
                items.append(s)
    return items

baskets = [row_to_basket(r) for r in df.itertuples(index=False, name=None)]
basket_sizes = pd.Series([len(b) for b in baskets], name="basket_size")

all_items = [item for b in baskets for item in b]
print("Total product appearances:", len(all_items))
print("Distinct products        :", len(set(all_items)))

# %% [markdown]
# ## 3. Descriptive statistics of basket size

# %%
stats = basket_sizes.describe()
print(stats)
print("Median:", basket_sizes.median())
print("Skewness:", round(basket_sizes.skew(), 3), "(positive = right-skewed)")

# %% [markdown]
# ## 4. Product frequency

# %%
counts = Counter(all_items)
freq = (pd.DataFrame(counts.items(), columns=["product", "baskets"])
          .sort_values("baskets", ascending=False)
          .reset_index(drop=True))
freq["share_of_baskets_%"] = (freq["baskets"] / len(baskets) * 100).round(1)
freq.index = freq.index + 1
print(freq.head(10))

top5 = freq["baskets"].head(5).sum()
print(f"\nTop-5 products: {top5} appearances = {top5 / len(all_items):.1%} of all entries")

# %%
top12 = freq.head(12).iloc[::-1]
plt.figure(figsize=(8, 5))
plt.barh(top12["product"], top12["baskets"], color="#6b8f7a")
plt.xlabel("Appearances")
plt.title("Most frequently listed products")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Basket-size distribution

# %%
size_counts = basket_sizes.value_counts().sort_index()
plt.figure(figsize=(8, 4))
size_counts.plot(kind="bar", color="#d9a58b")
plt.xlabel("Items per basket")
plt.ylabel("Transactions")
plt.title("Transactions by basket size")
plt.tight_layout()
plt.show()

print("Share of baskets with <= 3 items :", f"{(basket_sizes <= 3).mean():.1%}")
print("Share of baskets with > 10 items :", f"{(basket_sizes > 10).mean():.1%}")

# %% [markdown]
# ## 6. Feature engineering: binary basket vectors + PCA
# Each distinct product becomes one 0/1 feature (present / absent).

# %%
mlb = MultiLabelBinarizer()
X_full = mlb.fit_transform(baskets)
print("Binary matrix shape:", X_full.shape)
print("Sparsity (share of zeros):", f"{1 - X_full.mean():.2%}")

# Fixed random sample of up to 1,200 baskets for PCA and clustering
rng = np.random.RandomState(RANDOM_STATE)
n_sample = min(1200, X_full.shape[0])
idx = rng.choice(X_full.shape[0], size=n_sample, replace=False)
X = X_full[idx].astype(float)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
evr = pca.explained_variance_ratio_
print(f"PC1 = {evr[0]:.2%}, PC2 = {evr[1]:.2%}, total = {evr.sum():.2%}")

# %%
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=8, alpha=0.5, color="#6b8f7a")
plt.xlabel(f"PC1 ({evr[0]:.1%})")
plt.ylabel(f"PC2 ({evr[1]:.1%})")
plt.title("PCA of binary basket vectors (visual orientation only)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Clustering and evaluation
# Metrics: **silhouette** (higher is better, range -1..1) and **Davies-Bouldin** (lower is better).
# DBSCAN scores are computed on non-noise rows only, so they are not directly comparable.

# %%
results = []
labels_store = {}

def evaluate(name, labels, Xdata=X):
    labels = np.asarray(labels)
    mask = labels != -1                      # drop DBSCAN noise
    n_clusters = len(set(labels[mask]))
    n_noise = int((~mask).sum())
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = silhouette_score(Xdata[mask], labels[mask])
        dbi = davies_bouldin_score(Xdata[mask], labels[mask])
    else:
        sil, dbi = np.nan, np.nan
    results.append({"Algorithm": name, "Clusters": n_clusters, "Noise rows": n_noise,
                    "Silhouette": round(sil, 4), "Davies-Bouldin": round(dbi, 4)})
    labels_store[name] = labels

# K-means
km = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
evaluate("K-means", km.fit_predict(X))

# Hierarchical (Ward linkage)
hc = AgglomerativeClustering(n_clusters=4, linkage="ward")
evaluate("Hierarchical", hc.fit_predict(X))

# DBSCAN with Jaccard distance on boolean vectors
db = DBSCAN(eps=0.5, min_samples=5, metric="jaccard")
evaluate("DBSCAN", db.fit_predict(X.astype(bool)))

# Spectral clustering with nearest-neighbour affinity
sp = SpectralClustering(n_clusters=4, affinity="nearest_neighbors",
                        random_state=RANDOM_STATE, assign_labels="kmeans")
evaluate("Spectral", sp.fit_predict(X))

results_df = pd.DataFrame(results)
results_df

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, lab) in zip(axes.ravel(), labels_store.items()):
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=lab, s=8, cmap="tab10")
    ax.set_title(name)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Interpretation and limitations
# - A few staple products dominate and baskets are small and right-skewed.
# - Low or negative silhouette values mean the methods do not find well-separated groups
#   (expected for sparse, high-dimensional purchase data).
# - DBSCAN looks best only because it discards many baskets as noise and scores the rest.
# - No customer IDs, prices, quantities or dates, so no conclusions about revenue,
#   seasonality or causes. Next step: association-rule mining (support, confidence, lift).
