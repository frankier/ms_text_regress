# Datasets

There are some openly available datasets for ordinal regression of text on
Huggingface Hub, which can be loaded by `bert_ordinal`. Currently the datasets
are:

 * [`shoe_reviews`](https://huggingface.co/datasets/juliensimon/amazon-shoe-reviews):
   A single 5-point scale dataset of shoe reviews from Amazon.
 * [`cross_domain_reviews`](https://huggingface.co/datasets/frankier/cross_domain_reviews):
   A multi-scale dataset made by combining many review datasets from different
   domains.
 * [`multiscale_rt_critics`](https://huggingface.co/datasets/frankier/multiscale_rt_critics):
   A multi-scale dataset of different film critic review summaries along with
   their ratings as aggregated by RottenTomotoes.com

The function to load them is:

.. autofunction:: bert_ordinal.datasets.load_data
