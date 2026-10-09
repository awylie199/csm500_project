# Exploring Planning Decision Consistency Through Machine Learning Analysis of UK Open Government Data

## Abstract

The property planning domain has continued to receive significant interest for the application of machine learning. New
datasets and methods, particularly within Natural Language Processing, are enabling greater opportunities to analyse and
evaluate. England’s planning system is idiosyncratic in its discretionary nature: decisions ultimately depend on the
reviewing authority, introducing an element of ostensible variability. Furthermore, there has been significant policy
and systemic change over recent years.

This project utilises relatively new governmental datasets to explore patterns and identify potential misclassifications
through the application of traditional machine learning models and statistical analysis. Two datasets are created and
trialled simultaneously to capture the effects of physical property features. Inherent problems within the aggregated
datasets are outlined and addressed with preprocessing and feature engineering. A range of models are trialled with
results comparison. Ultimately an ensemble pipeline is outlined with thorough evaluation of its results within the
context of the project.

Ultimately SHAP analysis results demonstrate the benefit of deep learning techniques within the field through the
application of Sentence Bidirectional Encoder Representations from Transformers on short form text. Results are
qualified by diminishing performance on the minority classes despite methods taken to redress the target category
imbalance, as well as the highly flawed dataset, with limitations for wider application.

With further dataset improvements, this model shows the potential in applying traditional machine learning techniques as
the basis for evaluating application results over time, across councils and as a response to government policy.

## Code

```commandline
   uv run feature_development.ipynb
   uv run model_development.ipynb
```