"""

datasets package:
======================

TextAttack allows users to provide their own dataset or load from HuggingFace.


"""

from .dataset import Dataset
from .huggingface_dataset import HuggingFaceDataset
from .manifest import (
    ManifestDatasetView,
    SampleManifest,
    jointly_correct_indices,
    select_manifest_indices,
)

from . import helpers
