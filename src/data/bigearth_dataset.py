import pandas as pd
import os
from torch import Tensor
from torchgeo.datasets.bigearthnet import BigEarthNet
from typing import Callable, Dict, List, Optional
# from typing_extensions import override


class BigEarthDataset(BigEarthNet):

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(
                root,
                split,
                bands,
                num_classes,
                transforms,
                download,
                checksum,
                )

    # @override
    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        keys = ["s1", "s2"] if self.bands == "all" else [self.bands]
        urls = [self.metadata[k]["url"] for k in keys]
        md5s = [self.metadata[k]["md5"] for k in keys]
        filenames = [self.metadata[k]["filename"] for k in keys]
        directories = [self.metadata[k]["directory"] for k in keys]
        urls.extend([self.splits_metadata[k]["url"] for k in self.splits_metadata])
        md5s.extend([self.splits_metadata[k]["md5"] for k in self.splits_metadata])
        filenames_splits = [
            self.splits_metadata[k]["filename"] for k in self.splits_metadata
        ]

        "Add splits to be downloaded, only if directories dont exist, provides"
        "the pair images from s1 and s2"

        # Check if the split file already exist
        exists = []
        splits_exist = []
        for filename in filenames_splits:
            splits_exist.append(os.path.exists(os.path.join(self.root, filename)))

        master_exist = os.path.exists(os.path.join(self.root, 'master.csv'))

        if all(splits_exist) and not master_exist:
            self._combine_csvs(splits_exist)

        if master_exist:
            exists.append(master_exist)

        # Check if the files already exist
        for directory in directories:
            exists.append(os.path.exists(os.path.join(self.root, directory)))

        if all(exists):
            return



        # Check if zip file already exists (if so then extract)
        exists = []
        for filename in filenames:
            filepath = os.path.join(self.root, filename)
            if os.path.exists(filepath):
                exists.append(True)
                self._extract(filepath)
            else:
                exists.append(False)

        splits_exist = []
        for filename in filenames_splits:
            splits_exist.append(os.path.exists(os.path.join(self.root, filename)))

        if all(splits_exist) and not master_exist:
            self._combine_csvs(splits_exist)

        if master_exist:
            exists.append(master_exist)

        if all(exists):
            return


        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        filenames.extend(filenames_splits)

        # Download and extract the dataset
        for url, filename, md5 in zip(urls, filenames, md5s):
            self._download(url, filename, md5)
            filepath = os.path.join(self.root, filename)
            self._extract(filepath)

        splits_exist = []
        for filename in filenames_splits:
            splits_exist.append(os.path.exists(os.path.join(self.root, filename)))

        if all(splits_exist) and not master_exist:
            self._combine_csvs(splits_exist)


    # @override
    def _load_folders(self) -> List[Dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = 'master.csv'#self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata["s1"]["directory"]
        dir_s2 = self.metadata["s2"]["directory"]

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(",") for line in lines]

        folders = [
            {
                "s1": os.path.join(self.root, dir_s1, pair[1]),
                "s2": os.path.join(self.root, dir_s2, pair[0]),
            }
            for pair in pairs
        ]
        return folders

    def _combine_csvs(self, csvs: List) -> None:
        dfs = [pd.read_csv(file, header=None) for file in csvs]

        # ignore index stops 1st row being set as column names
        combined_df = pd.concat(dfs, ignore_index=True) 
        combined_df.to_csv(os.path.join(self.root, 'master.csv'), index=False)
