from torchgeo.datasets.utils import extract_archive
from tqdm import tqdm


srcs = ["/media/joshua/joshua/BigEarthNet-S1-v1.0.tar.gz", "/media/joshua/joshua/BigEarthNet-S2-v1.0.tar.gz"]
dst = "data"

for src in tqdm(srcs):
    extract_archive(src=src, dst=dst)
