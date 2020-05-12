# Toxicity Comments Experiment

This is the codes for the Toxicity Comments experiment in the paper. We mainly use the codes from [conversationai/unintended-ml-bias-analysis]( https://github.com/conversationai/unintended-ml-bias-analysis ) and nearly changed nothing, so we only upload the *make_weights.py* file and the weights we used here. 

It is also worth mentioning that the file *model_tool.py* seems a little problematic in their *master* branch, so we use that in the *AIES-2018* branch.

## Usage

```
# make weights
PYTHONHASHSEED=0 python make_weights.py
```
