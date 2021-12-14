## Re-CycleGT

### Dependencies
`pip install -r requirements.txt`

### How to Train/Evaluate
**Unsupervised Cycle Model** 

Perform the following to train and evaluate the unsupervised cycle model (which uses iterative back translation):

`python model.py`

**Supervised Text-to-Graph Model**


Perform the following to train and evaluate the supervised T2G model:

`python t2g_supervised.py`

**Supervised Graph-to-Text Model**

For training: `python g2t_supervised.py`

For evaluation: `python g2t_supervised_eval.py`

### How to Perform Inference

First, download the g2t model here and add it to your project directory: https://drive.google.com/file/d/1X_xx6wXt83EqZ5BveaHiIWPfcZhqAjsZ/view?usp=sharing Then, perform the following to get 5 random examples of G2T and 5 random examples of T2G from dev.json based on our final model:

`python inference.py`

This code can also be run after training, which downloads the best G2T and T2G models, and it will use those models for inference.

You can also manually input lists of dictionaries following the JSON object format described in the report (i.e. with 'entities' and 'relations' keys for graphs, or 'entities' and 'text' keys for text) in order to get predicted results from these inputs.


&nbsp;


WEBNLG 2017 Dataset: https://gitlab.com/shimorina/webnlg-dataset
