# External benchmark data

The Fodors-Zagats restaurant record-linkage tables used for the portfolio evaluation are available from the [DeepMatcher public dataset page](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md), which links to the University of Wisconsin-Madison-hosted [Fodors-Zagats files](https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/exp_data/).

The evaluator downloads only `tableA.csv` and `tableB.csv` when run with `--download`. The `class` column is used only to group matching records into entity-disjoint splits and derive match labels. It is never given to the classifier. The published random pair splits are deliberately not used because they reuse records across train, validation, and test.

The dataset is small and restaurant-specific. Its score is an offline benchmark result, not the Amazon challenge score and not a guarantee of performance on a new hackathon dataset. Check the dataset source's terms before redistributing the raw files.

## WDC Products feature experiment

The WDC Products archive (`80pair.zip`) is available from the [official WDC Products benchmark page](https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html), which describes pairwise train/validation/test sets, the 80%-corner-case condition, and test variants with 0%, 50%, or 100% unseen entities. The 100%-unseen gold file contains 4,500 pairs (500 positive, 4,000 negative). The project runner uses the `000un_train_large` pairs as a development pool, partitions the training pool by product cluster into cluster-disjoint train and validation groups, and tests on `100un_gs`. A check in the runner fails if a product cluster overlaps between the 000un development pool and the unseen test set. Cluster IDs are used for group splitting and labels only; they never enter model features.

Run `python portfolio/evaluate_wdc_products.py --download` from the project root. The archive is downloaded to this folder if it is not present. The baseline uses title, description, and brand lexical overlap/exactness. The engineered variant adds edit and character n-gram similarities, relative string lengths, title-number overlap/conflict, brand-token similarity, currency-aware price similarity (only comparable when the currencies match), currency conflict, and description presence. These are product-specific feature experiments, not changes to the trained Amazon business model. See `portfolio/wdc_products_evaluation.json` for thresholds, confusion matrices, and test metrics. The archive is excluded from `entity_resolution_portfolio.zip`; the runner can retrieve it when needed.

Results should be interpreted within this product benchmark. They measure transfer of a general pair-classification workflow and feature-engineering method, not accuracy on businesses, addresses, or the Amazon challenge.
