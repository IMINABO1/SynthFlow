import numpy as np
from pathlib import Path
from typing import Tuple
from numpy.typing import NDArray

DEFAULT_DATA_PATH = "dataset/BenchmarkDatasets/"


class DataSetLoader:
    """Loader for the FI-2010 benchmark (NoAuction, normalized).

    IMPORTANT structural facts (verified in data_analysis/eda.py):
    - Each .txt file is 149 feature-rows x N sample-columns; we keep the first
      40 rows (the raw LOB) and transpose to [N, 40].
    - The ``Training/Train_..._CF_k`` files are cumulative *per stock*: FI-2010
      concatenates 5 stocks, each with its own expanding window, so
      ``CF_k`` != ``[all day1 | all day2 | ...]``. Do NOT recover a day by
      prefix-slicing the training file.
    - The ``Testing/Test_..._CF_k`` file is a self-contained held-out day
      (day k+1). Use these for validation/test splits.
    """

    _N_FEATURES = 149
    _N_LOB_FEATURES = 40
    N_LEVELS = 10
    N_STOCKS = 5

    def __init__(self, data_path: str = DEFAULT_DATA_PATH,
                 normalization: str = "DecPre", use_cache: bool = True) -> None:
        self.data_path = Path(data_path)
        self.normalization = normalization  # ZScore, MinMax, or DecPre
        self.use_cache = use_cache

    def _get_file_path(self, day: int, dataset: str = "Training", auction: bool = False) -> Path:
        assert dataset in ("Training", "Testing")
        auction_str = "Auction" if auction else "NoAuction"
        subdir_num = {"ZScore": 1, "MinMax": 2, "DecPre": 3}[self.normalization]
        # On-disk quirk: the ZScore directory is spelled "Zscore".
        norm_dir = "Zscore" if self.normalization == "ZScore" else self.normalization
        subdir = f"{subdir_num}.{auction_str}_{norm_dir}"
        prefix = "Train" if dataset == "Training" else "Test"
        filename = f"{prefix}_Dst_{auction_str}_{self.normalization}_CF_{day}.txt"
        return (self.data_path / auction_str / subdir /
                f"{auction_str}_{self.normalization}_{dataset}" / filename)

    def load_file(self, day: int, dataset: str = "Training") -> NDArray[np.float32]:
        """Load one CF file's 40 LOB features as [n_samples, 40].

        Raw .txt files are large and ``np.loadtxt`` is slow, so the parsed array
        is cached to ``<file>.lob40.npy`` on first read.
        """
        filepath = self._get_file_path(day, dataset=dataset)
        cache_path = filepath.parent / f"{filepath.stem}.lob40.npy"

        if self.use_cache and cache_path.exists():
            return np.load(cache_path)

        data = np.loadtxt(filepath)
        lob_data = data[:self._N_LOB_FEATURES, :].T.astype(np.float32)

        if self.use_cache:
            try:
                np.save(cache_path, lob_data)
            except OSError:
                pass  # cache is an optimization only
        return lob_data

    def load_day(self, day: int) -> NDArray[np.float32]:
        """Backwards-compatible alias: the cumulative Training file CF_{day}."""
        return self.load_file(day, dataset="Training")

    def load_all_days(self) -> NDArray[np.float32]:
        """Full de-duplicated training corpus == the last cumulative file (CF_9)."""
        return self.load_file(9, dataset="Training")

    def load_split(self, train_fold: int = 7) -> Tuple[NDArray[np.float32],
                                                        NDArray[np.float32],
                                                        NDArray[np.float32]]:
        """Temporal train/val/test split using the benchmark's own files.

        With ``train_fold=7`` (default):
            train = Train_CF_7  (days 1-7, pooled)   ~254,750 rows
            val   = Test_CF_7   (day 8, held out)    ~ 55,478 rows
            test  = Test_CF_8   (day 9, held out)    ~ 52,172 rows
        All three are disjoint (no per-stock interleaving, no adjacency leakage).
        """
        train = self.load_file(train_fold, dataset="Training")
        val = self.load_file(train_fold, dataset="Testing")       # day train_fold+1
        test = self.load_file(train_fold + 1, dataset="Testing")  # day train_fold+2
        return train, val, test


def load_fi2010(
    data_path: str = DEFAULT_DATA_PATH,
    normalization: str = "DecPre",
    train_fold: int = 7,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Convenience entry point: returns (train, val, test) LOB arrays."""
    loader = DataSetLoader(data_path, normalization=normalization)
    return loader.load_split(train_fold=train_fold)


if __name__ == "__main__":
    datum = DataSetLoader(DEFAULT_DATA_PATH, normalization="DecPre")
    train, val, test = datum.load_split()
    print("train/val/test:", train.shape, val.shape, test.shape)
    print("dtypes:", train.dtype, val.dtype, test.dtype)
