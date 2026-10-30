"""Structural alerts from RDKit's FilterCatalog (Brenk, PAINS, NIH)."""
import numpy as np
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

CATALOGS = ("BRENK", "PAINS", "NIH")

def _catalog():
    p = FilterCatalogParams()
    for n in CATALOGS: p.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, n))
    return FilterCatalog(p)

def alert_matrix(smiles):
    """Return (A, names, atoms): A[i,j]=1 if alert j matches molecule i; atoms[i] = {alert_name: set(atom idx)}."""
    cat = _catalog(); names = {}; rows = []; atoms = []
    for s in smiles:
        m = Chem.MolFromSmiles(s); hit = {}; a = {}
        if m is not None:
            for e in cat.GetMatches(m):
                d = e.GetDescription()
                try: src = e.GetProp("FilterSet")
                except Exception: src = ""
                key = f"{src}:{d}" if src else d
                names.setdefault(key, len(names)); hit[key] = 1
                for fm in e.GetFilterMatches(m):
                    a.setdefault(key, set()).update(j for _, j in fm.atomPairs)
        rows.append(hit); atoms.append(a)
    ordered = sorted(names, key=names.get); A = np.zeros((len(smiles), len(ordered)), np.uint8)
    for i, h in enumerate(rows):
        for k in h: A[i, names[k]] = 1
    return A, ordered, atoms
