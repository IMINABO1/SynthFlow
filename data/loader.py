import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from numpy.typing import NDArray
from collections import defaultdict

class DataSetLoader:

    _N_FEATURES = 149
    _N_LOB_FEATURES = 40
    N_LEVELS = 10
    N_STOCKS = 5
    N_DAYS = 9
    shape_per_day = defaultdict(tuple)

    def __init__(self, data_path: str, normalization: str = "DecPre") -> None:
        self.data_path = Path(data_path)
        self.normalization = normalization
        # possible normaliations are ZScore, MinMax and DecPre

    def _get_file_path(self, day: int, auction: bool = False) -> Path:
        auction_str = "Auction" if auction else "NoAuction"

        if self.normalization == "ZScore":
            subdir = f"1.{auction_str}_ZScore"
            filename = f"Train_Dst_{auction_str}_{self.normalization}_CF_{day}.txt"
        elif self.normalization == "MinMax":
            subdir = f"2.{auction_str}_MinMax"
            filename = f"Train_Dst_{auction_str}_{self.normalization}_CF_{day}.txt"
        else:  # DecPre
            subdir = f"3.{auction_str}_DecPre"
            filename = f"Train_Dst_{auction_str}_{self.normalization}_CF_{day}.txt"
        
        return self.data_path / auction_str / subdir / f"{auction_str}_{self.normalization}_Training" / filename

    def load_day(self, day: int) -> NDArray[np.float32]:
        filepath = self._get_file_path(day)
        data = np.loadtxt(filepath)

        lob_data = data[:self._N_LOB_FEATURES, :].T
        return lob_data.astype(np.float32)
    
    """
    def load_all_days(self, days: Optional[List] = None) -> NDArray[np.float32]:
        if days == None or days == []:
            days = [*range(1, self.N_DAYS + 1)]
        print(f"Loading days: {days}")
        all_data = []
        
        for day in days:
            try:
                day_data = self.load_day(day)
                all_data.append(day_data)
                self.shape_per_day[day] = day_data.shape
                print(f"Day {day}: {len(day_data)} samples")
            except FileNotFoundError as e:
                print(f"Warning: Could not load day {day}: {e}")
            except Exception as e:
                print(f"Error loading day {day}: {e}")
        return np.concatenate(all_data, axis=0)
    """

    def load_all_days(self) -> NDArray[np.float32]:
        return self.load_day(9)

def load_fi2010(data_path: str) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    loader = DataSetLoader(data_path, normalization="DecPre")

    train_data = loader.load_all_days()



datum = DataSetLoader(data_path="../BenchmarkDatasets/", normalization="DecPre")
file = datum._get_file_path(1, False)

cf1 = datum.load_day(1)
cf2 = datum.load_day(2)

print(np.allclose(cf2[:39512], cf1))
print(np.allclose(cf2[:3455], cf1[:3455]))

# day_ = datum.load_day(1)
# print(day_)
# days = datum.load_all_days()
# print(datum.shape_per_day)

# print(file)
# print(file.exists())
