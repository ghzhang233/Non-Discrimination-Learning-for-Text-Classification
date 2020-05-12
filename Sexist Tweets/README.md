# Sexist Tweets Experiment

This is the codes for the Sexist Tweets experiment in the paper.

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
opencv-python==4.1.0.25
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

For <i>adjective_negative.txt, adjective_people.txt, adjectives_postive.txt, filler.txt, verbs_negative.txt, verbs_positive.txt</i>, we download from [conversationai/unintended-ml-bias-analysis]( https://github.com/conversationai/unintended-ml-bias-analysis ).

For *gender_extra_swap_words.txt, gender_general_swap_words.txt*, we download from [uclanlp](https://github.com/uclanlp)/**[corefBias](https://github.com/uclanlp/corefBias)** and remove duplicate by hand.

For *st.csv*, we download from [ZeerakW](https://github.com/ZeerakW)/**[hatespeech](https://github.com/ZeerakW/hatespeech)** and scratch the tweets from Twitter. We record the ids we uesd in our paper as *st_used_id.csv*.

For *glove.840B.300d.txt*, we download from [ GloVe: Global Vectors for Word Representation ]( https://nlp.stanford.edu/projects/glove/ ).

For *GoogleNews-vectors-negative300-hard-debiased.txt*, we download from [tolga-b](https://github.com/tolga-b)/**[debiaswe](https://github.com/tolga-b/debiaswe)**.

```
./
│  config.py
│  data_utils.py
│  general_utils.py
│  main.py
│  make_madlib.py
│  make_weights.py
│  models.py
|  requirements.txt
├─data
│      st.csv
|      st_used_id.csv
│      adjectives_negative.txt
│      adjectives_people.txt
│      adjectives_positive.txt
│      filler.txt
│      gender_extra_swap_words.txt
│      gender_general_swap_words.txt
│      verbs_negative.txt
│      verbs_positive.txt
|      glove.840B.300d.txt
|      GoogleNews-vectors-negative300-hard-debiased.txt
├─processed_data
│      madlib.csv
|      weights_st.npy
```

## Usage

```
# make data
PYTHONHASHSEED=0 python data_utils.py

# make IPTTS
PYTHONHASHSEED=0 python make_madlib.py

# make weights
PYTHONHASHSEED=0 python make_weights.py

# for baseline
PYTHONHASHSEED=0 python main.py --round 10

# for augmentation
PYTHONHASHSEED=0 python main.py --use_augmentation --round 10

# for non-discrimination learning
PYTHONHASHSEED=0 python main.py --use_weights --round 10
```

We note that make *make_madlib.py* is partly from [conversationai/unintended-ml-bias-analysis]( https://github.com/conversationai/unintended-ml-bias-analysis ).