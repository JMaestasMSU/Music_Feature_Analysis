Jarred Maestas

CS 3120

**ML Project Proposal: Music Feature Analysis and Genre Classification**

**Option B: Data Exploration and Modeling**

**Dataset Description**

I will use the Free Music Archive (FMA) dataset containing 8,000 30-second audio clips across 8
genres with pre-computed audio features (MFCCs, spectral features) and metadata. This interests
me because music data combines complex temporal and frequency patterns, offering unique
challenges beyond standard datasets while building on my experience with time-series analysis.

**Exploratory Data Analysis**

I plan to extract and analyze audio features including MFCCs, spectral centroids, tempo, and chroma
vectors. Visualizations will include spectrograms, t-SNE projections showing genre clustering, and
correlation matrices between audio features and popularity metrics. I'll examine how different audio
characteristics distinguish musical genres.

**Modeling Approach**

**Baseline Model:** Random Forest classifier for multi-class genre classification using extracted audio
features to establish performance benchmarks.

**Advanced Models:** (1) CNN processing raw spectrograms for end-to-end learning, comparing
learned vs hand-crafted features; (2) Autoencoder for unsupervised feature learning combined with K-
means clustering to discover music patterns, using cluster assignments to enhance supervised
classification.

**Model Validation**

Stratified 5-fold cross-validation for genre classification with custom metrics including genre-weighted
F1 scores. Will compare model performance across different feature representations (raw audio,
hand-crafted features, learned embeddings).

**Tools and Technologies**

Python libraries: librosa (audio processing), scikit-learn (ML models), PyTorch (deep learning),
pandas/numpy (data manipulation), matplotlib/seaborn (visualization), and umap-learn
(dimensionality reduction).

**Interest/Motivation**

This project excites me because it applies ML to creative domains, combining my passion for music
with technical challenges in feature engineering, dimensionality reduction, and multi-modal learning.
Unlike typical classification tasks, audio analysis requires specialized preprocessing and offers
opportunities to compare traditional ML with deep learning approaches on complex, high-dimensional
data.