"""Extremophilic Protein Translator (eptrans).

A pipeline to build a labeled dataset of secreted proteins from GTDB
extremophiles, for fine-tuning a protein language model / classifier.

Stages
------
1. gtdb        - index GTDB r232 metadata + on-disk genome/proteome accessors
2. binning     - metadata keyword flags + GenomeSPOT predictions -> environment labels
3. reconcile   - reuse published GenomeSPOT predictions; compute r232 delta
4. selection   - phylogenetically-controlled extremophile / mesophile-outgroup picks
5. signalp     - signal-peptide prediction -> secreted (mature) proteins
6. dataset     - assemble labeled, leakage-aware train/val/test dataset
"""

__version__ = "0.1.0"

from . import config  # noqa: F401

__all__ = ["config"]
