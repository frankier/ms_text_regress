.. bert_ordinal documentation master file, created by
   sphinx-quickstart on Fri Oct 21 21:08:30 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bert_ordinal's documentation!
========================================

bert_ordinal is a Python package for ordinal regression using BERT. The main
content of the package BERT compatible models based on `transformers`, ready to
fine-tune on ordinal regression tasks as well as generally applicable functions
for working with ordinal data. Apart from this, functions for evaluating the
task and utilities for fetching and processing  some open datasets for ordinal
regression are provided.

Modules
-------

.. currentmodule:: bert_ordinal

.. autosummary::
   :toctree: _autosummary
   :recursive:

   ordinal_models.bert
   datasets
   ordinal
   eval

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   datasets

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
