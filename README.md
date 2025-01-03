<h3 align="center">
<p> Predictor of sumoylation sites</p> </h3>

The repository contains scripts for saving built and trained ISUMsite1 models as well as predictor scripts. And a predictor script built using the model.

### Dependency

```
python                  2.7.15
codecs              
csv             
pickle            
Tkinter      
tkFileDialog           
tkMessageBox        
os         
pandas                  0.24.2
scipy                   1.2.3
numpy                   1.16.6
sklearn-learn           1.3.2        
```

### Feature extraction
Pse-in-One-2.0           http://bliulab.net/Pse-in-One2.0/download/

### Dataset

Dataset 1
Dataset 2

### Train and Test

#### Train

Save the model Model1 built on dataset 1

```python
python Train_Model1.py
```

Save the model Model2 built on dataset 2

```python
python Train_Model2.py
```

#### Test

run predictor of sumoylation sites

```shell
python Visualization_Predictor.py
```
