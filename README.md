# NLP project - Offensive language exploratory analysis

## Description

This is repository for project in Natural Language Processing course at the University of Ljubljana 2020/21. 

## Dataset acquistion (and merging)
Use full_dataset_all_labels.csv for any analysis. It is a merged dataset from different sources (listed in report).


## Repository organisation
** in folder "datasets" there are all selected datasets, 
** in folder images are all images that were generated during project work, not all of them were used in the report
** NLP_project.pdf is a pdf file with final report for project 
** "full_dataset_all_labels.csv" is a csv file with all datasets merged. This file was then used for further work
** "bert_elmo.ipynb" is a jupyter notebook in which the contextual embeddings were produced
** "statistics.ipynb" is a jupyter notebook which was intended for producing some statistics about datesets, but there is not much work done in it, 
** "process_datasets.ipynb" is a jupyter notebook in which datasets were preprocessed and then mergen into final one, 
** "analysis_jus.ipynb" is a jupyter notebook in which non-contextual emmbedings (TF-IDF, word2vec, GLOve and fastText) and topic modelling were done,
** "upset.ipynb" is a jupyter notebook in which upset plots were created, there is also some word done with TF-IDF representations.

## Reproducibility
We used Anaconda environement so our code can be easily rerun. Requirements for recreating environment and thus easily running the notebooks are written in "environment.yml" file.
