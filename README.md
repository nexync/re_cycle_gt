## Re-CycleGT

### Dependencies
`pip install -r requirements.txt`

### How to Run
**Unsupervised Cycle Model** 

Perform the following to train and evaluate the unsupervised cycle model (which uses iterative back translation):

`python model.py`

**Supervised Text-to-Graph Model**


Perform the following to train and evaluate the supervised T2G model (which uses iterative back translation):

`python t2g_supervised.py`

**Supervised Graph-to-Text Model**

For training: `python g2t_supervised.py`

For evaluation: `python g2t_supervised_eval.py`

&nbsp;

WEBNLG 2017 Dataset: https://gitlab.com/shimorina/webnlg-dataset
