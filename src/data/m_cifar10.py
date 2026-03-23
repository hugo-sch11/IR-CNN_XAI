"""
Docstring for m_data.cifar10
Program that process (manually) cifar-10 ~ original batches for python.
>(useless with torchvision.datasets.CIFAR10)
"""

import pickle
import warnings
from numpy.exceptions import VisibleDeprecationWarning
from typing import Dict, List
import os

# cifar-10 used a deprecated numpy version
warnings.filterwarnings(
    "ignore",
    category=VisibleDeprecationWarning,
)

"""
cifar-10 documentation code
"""
def unpickle(file: str) -> Dict:
    with open(file, 'rb') as fo:
        dict: Dict = pickle.load(fo, encoding='bytes')
    return dict

DIRECTORY = "cifar-10-batches-py"
data_batches: List[Dict] = []
for file in os.listdir(DIRECTORY):
    path: str = os.path.join(DIRECTORY, file)
    if os.path.isfile(path):
        batch: Dict = unpickle(path)
        data_batches.append(batch)

### => dict_keys([b'batch_label', b'labels', b'data', b'filenames']) / dict_keys([b'num_cases_per_batch', b'label_names', b'num_vis'])
# for batch in data_batches:
#     print(batch.keys())

for i in range(data_batches.__len__()):
    print(f"batch {i} keys : {data_batches[i].keys()}")

# print(f"{data_batches[0].values()}")
# print(data_batches[0].get(b"data"))