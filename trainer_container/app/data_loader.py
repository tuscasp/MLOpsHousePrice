from pathlib import Path


from typing import override

import pandas as pd


class SupervisedDatasetLoader:
    def __init__():
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class CsvLoader(SupervisedDatasetLoader):

    def __init__(self, path: Path, target_column):
        
        self._pathDataFile = Path(path)
        if (not self._pathDataFile.exists()):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self._targetColumn = target_column

    @override
    def load(self):
        full_data = pd.read_csv(self._pathDataFile)
        feature_columns = [c for c in full_data.columns if not c == self._targetColumn]
        target = full_data[self._targetColumn]
        features = full_data[feature_columns]

        return features, target

