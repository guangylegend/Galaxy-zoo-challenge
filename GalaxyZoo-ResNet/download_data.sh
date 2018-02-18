#!/bin/bash
# Use my cookie to download the data needed

wget -x -c --load-cookies cookie.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/download/images_test_rev1.zip
wget -x -c --load-cookies cookie.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/download/images_training_rev1.zip
wget -x -c --load-cookies cookie.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/download/training_solutions_rev1.zip

cd data
unzip -qq images_test_rev1.zip
unzip -qq images_training_rev1.zip
unzip -qq training_solutions_rev1.zip
