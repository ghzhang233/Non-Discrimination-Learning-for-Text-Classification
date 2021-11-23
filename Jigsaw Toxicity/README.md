# Jigsaw Toxicity Experiment

This is the codes for the Jiwsaw Toxicity experiment in the paper.

## Environment

We generate this *requirements.txt* by *pip freeze* command automatically. If there are any problems with this, please feel free to contact me for help.

```
absl-py==0.7.0
astor==0.7.1
certifi==2019.3.9
cffi==1.12.1
cycler==0.10.0
Cython==0.29.6
et-xmlfile==1.0.1
fasttext==0.8.3
future==0.17.1
gast==0.2.2
grpcio==1.19.0
h5py==2.9.0
jdcal==1.4
json-lines==0.5.0
Keras==2.2.4
Keras-Applications==1.0.7
Keras-Preprocessing==1.0.9
kiwisolver==1.1.0
Markdown==3.0.1
matplotlib==3.1.1
mkl-fft==1.0.10
mkl-random==1.0.2
mock==2.0.0
nltk==3.4.1
numpy==1.16.2
openpyxl==2.6.2
pandas==0.24.1
pbr==5.1.3
Pillow==6.1.0
protobuf==3.7.0
pybind11==2.2.4
pycparser==2.19
pyparsing==2.4.2
python-dateutil==2.8.0
pytz==2018.9
PyYAML==3.13
scikit-learn==0.20.3
scipy==1.2.1
six==1.12.0
tensorboard==1.13.1
tensorflow==1.12.0
tensorflow-estimator==1.13.0
tensorflow-gpu==1.13.1
termcolor==1.1.0
torch==1.0.1
torchvision==0.2.1
tqdm==4.33.0
Werkzeug==0.14.1
```

## Data

You should download the data and merge them as listed below.

For <i>adjective_people.txt, bias_madlibs_77k.csv, wiki_debias_train.csv, wiki_debias_dev.csv</i>, we download from [conversationai/unintended-ml-bias-analysis]( https://github.com/conversationai/unintended-ml-bias-analysis ).

For *glove.840B.300d.txt*, we download from [ GloVe: Global Vectors for Word Representation ]( https://nlp.stanford.edu/projects/glove/ ).

For *train.csv, test.csv*, we download from [Jigsaw Unintended Bias in Toxicity Classification]( https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification ).

```
./
│  config.py
│  data_utils.py
│  general_utils.py
│  get_proportion.py
│  main.py
│  make_weights.py
│  models.py
│
├─data
│      adjectives_people.txt
│      bias_madlibs_77k.csv
│      glove.840B.300d.txt
│      identity_columns.txt
│      test.csv
│      train.csv
│      wiki_debias_dev.csv
│      wiki_debias_train.csv
│
├─processed_data
│      weights.npy
```

## Usage

```
# make data
PYTHONHASHSEED=0 python data_utils.py

# make weights
PYTHONHASHSEED=0 python make_weights.py

# for baseline
PYTHONHASHSEED=0 python main.py --round 10 --name_model biased

# for supplementation
PYTHONHASHSEED=0 python main.py --use_supplementation --round 10 --name_model sup

# for non-discrimination learning
PYTHONHASHSEED=0 python main.py --use_weights --round 10 --name_model weight

# For generate Tabel 8 as shown in the paper
PYTHONHASHSEED=0 python get_proportion.py
```

It is worth mentioning that in this experiments, as we set labels as *0-BAD* and *1-NOT-BAD*. As we always set *Offensive* as 1 in the paper,  we report EFPR and EFNR conversly in Tabel 5. 
