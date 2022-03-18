<h1><strong>Cross-validated DEA (CVDEA) </strong></h1>

<h2>Installation</h2>
<p>To facilitate installation on a personal computer, we recommend installing git (see: https://git-scm.com/downloads) and the Anaconda distribution (see: https://www.anaconda.com/products/individual). The steps to follow are based on these two installations.</p>

<b>Step1</b>. Open the Anaconda Prompt console, place it in the desired directory for installation and enter the instruction: 
```
https://github.com/MiriamEsteve/CVDEA.git
```

<b>Step 2</b>. Place us in the folder created by CVDEA, using "cd CVDEA", and execute the instruction:
```
python setup.py install
```
<p>By following these two simple steps we will have the CVDEA package installed on our computer. The examples included below are also available in the CVDEA/test folder that has been created in the installation.</p>


<h2>Import libraries</h2>
<p>All the libraries in the repository are imported since they will be used in all the examples presented.</p>

```python
import numpy as np
import random
import cvdea
```

<h2>Generate MONO OUTPUT simulated data </h2>
<p>CVDEA repository includes a simulated data generator module. It is used as an example of the use of the repository. For do that, the seed of the generator and the size of the dataset are stablished.</p>

```python
N = 50
sd = random.randint(0, 1000)
matrix = cvdea.Data(sd, N, 1).data
```

<p>This dataset include the theorical output. In this sense, it is necessary save it in a differet variable.</p>

```python
matrixTheoric = matrix.copy()
matrix = matrix.iloc[:, :-1]
```

<h2>Create the CVDEA model</h2>
<p>The creation of the CVDEA model consist on specify the inputs and outputs columns name (x and y), the depth configuration, i.e. the number of hyperplanes in each combination of supporting hyperplanes (D) and the number of different combinations of supporting hyperplanes (P). Once this is done, the model is created and fitted to build it.</p>

<b>Step 1. </b> The name of the columns of the inputs and outputs in the dataset are indicated. If these ones don't exist in the dataset, the CVDEA model returns an error. 
```python
y = [matrix.columns[-1]]
x = list(matrix.drop(y, axis=1).columns)
```

<b>Step 2.</b> the depth configuration, i.e. the number of hyperplanes in each combination of supporting hyperplanes (D) and the number of different combinations of supporting hyperplanes (P) are specified.
```python
D = 10
P = 10
```
<b>Step 3.</b> The creation and fit of the CVDEA model are done.
```python
model = cvdea.CVDEA(matrix, x, y, D, P)
model.fit_CVDEA()
```

<h2>Predictions</h2>
<p>The prediction of the CVDEA model can be with one dataset or with a single register of the dataset. 

```python
matrix_pred = model.prediction(matrix)
```
