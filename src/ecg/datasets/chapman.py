"""Chapman-Shaoxing (and Ningbo) reader; same WFDB/SNOMED format, one record file per patient."""
from ecg.datasets.cinc import iter_wfdb_dx


def iter_chapman(root, limit: int | None = None):
    yield from iter_wfdb_dx(root, "chapman", limit, patient_from_record=True)


def iter_ningbo(root, limit: int | None = None):
    yield from iter_wfdb_dx(root, "ningbo", limit, patient_from_record=True)
