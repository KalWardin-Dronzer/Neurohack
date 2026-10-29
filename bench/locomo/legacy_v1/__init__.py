"""Frozen copy of the v1.0-hackathon memory code, kept only as a benchmark baseline.

The files are `git show v1.0-hackathon:app/memory/<name>.py` and `app/config.py`, unchanged except that
retriever.py imports Settings from this package. Its bugs (such as the inverted BM25 normalization) are
deliberately kept so that the baseline measures what was actually submitted.
"""
