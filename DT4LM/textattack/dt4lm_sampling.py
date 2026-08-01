"""Package-local access to DT4LM's pure sampling helpers.

PEP 660 editable installs can expose ``textattack`` without exposing sibling
top-level modules. The fallback loads the source file directly so existing
editable environments work before they are reinstalled.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


try:
    from dt4lm_sampling import (  # type: ignore
        SAMPLING_ALGORITHM_ALL,
        SAMPLING_ALGORITHM_HASH,
        select_sample_indices,
        selection_hash,
        validate_sample_manifest_payload,
    )
except ModuleNotFoundError:
    source = Path(__file__).resolve().parents[1] / "dt4lm_sampling.py"
    spec = spec_from_file_location("_dt4lm_sampling_source", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DT4LM sampling helpers from {source}.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    SAMPLING_ALGORITHM_ALL = module.SAMPLING_ALGORITHM_ALL
    SAMPLING_ALGORITHM_HASH = module.SAMPLING_ALGORITHM_HASH
    select_sample_indices = module.select_sample_indices
    selection_hash = module.selection_hash
    validate_sample_manifest_payload = module.validate_sample_manifest_payload


__all__ = [
    "SAMPLING_ALGORITHM_ALL",
    "SAMPLING_ALGORITHM_HASH",
    "select_sample_indices",
    "selection_hash",
    "validate_sample_manifest_payload",
]
