**AI Recommendation System**

This repository contains a student-focused, neural-network-based recommendation system implemented in Python using PyTorch's `torch.nn` modules. The project is designed as a clear, extensible foundation for building and experimenting with modern collaborative and hybrid recommendation models.

**Project Highlights:**
- **Purpose:**: Train and evaluate neural recommender models on small-to-medium sized movie rating datasets for learning and experimentation.
- **Frameworks:**: Python and PyTorch (`torch.nn`) are used for model implementation and training.
- **Student-friendly:**: Structured for readability, reproducibility, and easy extension — the kind of project a student can be proud of.

**Dataset**
- **Source:**: The repository includes the MovieLens `ml-latest-small` dataset under `ML_DATA/ml-latest-small/`.
- **Files included:**: `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`.
- **Summary (from dataset README):**: ~100,836 ratings, 3,683 tag applications, 9,742 movies, 610 users. Ratings use a 0.5–5.0 star scale and are timestamped.
- **Format:**: CSV with headers. `ratings.csv` rows are `userId,movieId,rating,timestamp`. `movies.csv` rows are `movieId,title,genres`.

**Recommended Project Layout**
- **Data:**: `ML_DATA/ml-latest-small/` (already present).
- **Code (suggested):**: `data/` (loaders, preprocessors), `models/` (PyTorch modules), `train.py`, `evaluate.py`, `utils.py`.

**Dependencies**
- **Language:**: Python 3.8+.
- **Key packages:**: `torch` (PyTorch), `pandas`, `numpy`. Optionally `scikit-learn` for metrics and splitting, and `tqdm` for progress bars.
- **Install:**: install PyTorch following the official instructions and then `pip install pandas numpy` (or use a `requirements.txt` you create).

**Overview of the Approach**
- **Modeling Philosophy:**: Use embedding layers for categorical IDs (users/movies), combine with optional side information (movie genres, tags) and feed into one or more neural layers built with `torch.nn` to predict ratings or ranking scores.
- **Typical Components:**: ID embeddings, feature encoders, a small fully-connected network, and a prediction head. Loss functions commonly used: regression losses (MSE) for rating prediction or pairwise/ranking losses for ranking tasks.
- **Evaluation:**: Use train/validation/test splits (or cross-validation). Report RMSE for rating prediction and ranking metrics (e.g., precision@k, recall@k, NDCG) for top-K recommendations.

**Getting Started (student quick-run)**
1. Place or confirm the dataset is available at `ML_DATA/ml-latest-small/` (already included in this workspace).
2. Create virtual environment: `python -m venv .venv && source .venv/bin/activate`.
3. Install dependencies: `pip install torch pandas numpy` (adjust PyTorch install for CUDA/CPU as needed).
4. Implement or run training: `python train.py --data-dir ML_DATA/ml-latest-small/` (create `train.py` following the suggested layout).

**Notes & Best Practices**
- **Reproducibility:**: Fix random seeds (PyTorch, NumPy, Python `random`) and log hyperparameters and results.
- **Preprocessing:**: Parse movie genres carefully (pipe-separated), and consider mapping timestamps to features (year, recency) if helpful.
- **Baselines:**: Start with simple matrix-factorization or user/item average baselines before moving to neural models.

**Next Steps (suggested for a student)**
- Implement `data/` loaders that produce PyTorch `Dataset`/`DataLoader` objects.
- Implement a small neural collaborative filtering model in `models/` using `torch.nn.Embedding` and MLP layers.
- Add `train.py` and `evaluate.py` with clear CLI arguments and logging.
- Experiment with hybrid inputs (genres/tags) and compare to simple baselines.

**Contributing & License**
- This is a student project scaffold — adapt, extend, and learn. If you reuse data or publish results, follow the MovieLens dataset citation and license terms included in `ML_DATA/ml-latest-small/README.txt`.

If you want, I can now scaffold `train.py`, `models/`, and data loaders next — tell me which piece to generate first.

