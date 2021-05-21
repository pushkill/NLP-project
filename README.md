# NLP project - Offensive language exploratory analysis

## Description

This is repository for project in Natural Language Processing course at the University of Ljubljana 2020/21. 

## Dataset acquistion (and merging)
Use full_dataset_all_labels.csv for any analysis. It is a merged dataset from all sources that we used. This contains 2 columns: text and label.
We have listed and described the datasets we used in the report (as well as cited the datasets' authors).  
Here we where said datasets can be found
* Kaggle dataset can be found at https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
* Davidsons dataset can be found at https://github.com/t-davidson/hate-speech-and-offensive-language
* Vidgens dataset can be found at https://www.kaggle.com/usharengaraju/dynamically-generated-hate-speech-dataset?select=2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv
* Mandls dataset can be found at https://hasocfire.github.io/hasoc/2019/dataset.html 
* Waseems dataset can be found at https://github.com/DataforGoodIsrael/DetectHateSpeech
* Chandrasekharans dataset can be found at https://github.com/ceshwar/reddit-norm-violations
* Cacholas dataset can be found at https://github.com/ericholgate/vulgartwitter
* Formspring dataset can be found at https://github.com/StefanoFrazzetto/CrimeDetector/tree/master/datasets/formspring but was originally provided at https://www.kaggle.com/swetaagrawal/formspring-data-for-cyberbullying-detection
* dataset by Golbeck et al. can be obtained by asking them via e-mail (provided in their paper)


## Repository organisation
* in folder "datasets" there are all selected datasets, 
* in folder images are all images that were generated during project work, not all of them were used in the report
* NLP_project.pdf is a pdf file with final report for project 
* "full_dataset_all_labels.csv" is a csv file with all datasets merged. This file was then used for further work
* "bert_elmo.ipynb" is a jupyter notebook in which the contextual embeddings were produced
* "statistics.ipynb" is a jupyter notebook which was intended for producing some statistics about datesets, but there is not much work done in it, 
* "process_datasets.ipynb" is a jupyter notebook in which datasets were preprocessed and then mergen into final one, 
* "analysis_jus.ipynb" is a jupyter notebook in which non-contextual emmbedings (TF-IDF, word2vec, GLOve and fastText) and topic modelling were done,
* "upset.ipynb" is a jupyter notebook in which upset plots were created, there is also some word done with TF-IDF representations.

## Reproducibility
We used Anaconda environement so our code can be easily rerun. Requirements for recreating environment and thus easily running the notebooks are written in "environment.yml" file.
