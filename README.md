<h1><strong>Efficiency Analysis Trees Boost (EATBoost)</strong></h1>

<p style="justify">EAT is a new methodology based on regression trees for estimating production frontiers satisfying fundamental postulates of microeconomics, such as free disposability. This new approach, baptized as Efficiency Analysis Trees (EAT), shares some similarities with the Free Disposal Hull (FDH) technique. However, and in contrast to FDH, EAT overcomes the problem of overfitting by using cross-validation to prune back the deep tree obtained in the first stage. Finally, the performance of EAT is measured via Monte Carlo simulations, showing that the new approach reduces the mean squared error associated with the estimation of the true frontier by between 13% and 70% in comparison with the standard FDH.</p>

For more info see: https://doi.org/10.1016/j.eswa.2020.113783

<h2>Installation</h2>
<p>To facilitate installation on a personal computer, we recommend installing git (see: https://git-scm.com/downloads) and the Anaconda distribution (see: https://www.anaconda.com/products/individual). The steps to follow are based on these two installations.</p>

<b>Step1</b>. Open the Anaconda Prompt console, place it in the desired directory for installation and enter the instruction: 
```
git clone https://github.com/MiriamEsteve/EATBoost.git
```

<b>Step 2</b>. Place us in the folder created by EATpy, using "cd EATpy", and execute the instruction:
```
python setup.py install
```
<p>By following these two simple steps we will have the EATpy package installed on our computer. The examples included below are also available in the EATpy/test folder that has been created in the installation.</p>

<h2>Read Data sets</h2>
<p>The input data set that requires all the functionalities that will be explained later in this section must meet the condition of being a DataFrame data type from the pandas library. For this purpose, the user can start from an Excel document in which there are as many columns as inputs and as many columns as outputs. In all of them a header with their names must be included, because it will be necessary to introduce them in the EATpy package. The user can also start from a CSV file in which the specification of the data headers will be necessary.</p>

<p>Once the type of file chosen complies with the indicated structure, it must be converted to DataFrame type from the pandas library. The instructions are:</p>

- Import the pandas library
```python
import pandas as pd
```

- Reading and converting an Excel document
```python
dataset = pd.read_excel("file_name.xlsx", sheet_name = "data")
```

- Reading and converting a CSV file
```python
dataset = pd.read_csv("file_name.csv", sep = ";")
```

<p>Once these steps have been completed, you can start using the EATpy library.</p>

<h2>Import libraries</h2>
<p>All the libraries in the repository are imported since they will be used in all the examples presented.</p>

```python
import eat
import graphviz
```

<h2>Generate simulated data </h2>
<p>EATpy repository includes a simulated data generator module. It is used as an example of the use of the repository. For do that, the seed of the generator and the size of the dataset are stablished.</p>

```python
dataset = eat.Data(1, 50).data
```
<h2>Create the EAT model</h2>
<p>The creation of the EAT model consist on specify the inputs and outputs columns name, the ending rule and the number of folder for Cross-Validation process. Once this is done, the model is created and fitted to build it.</p>

<b>Step 1. </b> The name of the columns of the inputs and outputs in the dataset are indicated. If these ones don't exist in the dataset, the EAT model returns an error. 
```python
x = ["x1", "x2"]
y = ["y1", "y2"]
```

<b>Step 2.</b> The ending rule and the number of fold in Cross-Validation process are specified.
```python
numStop = 5
fold = 5
```
<b>Step 3.</b> The creation and fit of the EAT model are done.
```python
model = eat.EAT(dataset, x, y, numStop, fold)
model.fit()
```

<h2>Draw tree EAT</h2>
<p>The drawing of the EAT tree is done using the external graphviz library. For this purpose, the first instruction generates the dot_data that graphviz needs to draw the EAT tree. In addition, it saves it as an image in the working directory.</p>

```python
dot_data = model.export_graphviz('EAT')
graph = graphviz.Source(dot_data, filename="tree", format="png")
graph.view()
```

<h2>Predictions</h2>
<p>The prediction of the EAT model can be with one dataset or with a single register of the dataset. To do this, you need the data set or single register you want to predict and the names of the input columns. In order to indicate the names of the inputs in the dataset to be predicted. As a general rule, these names will be the same as those in the initial dataset.</p>
<p>In this example, the first 10 register are selected from the initial dataset and the name of the inputs are the same as it. Then, the model EAT realize the prediction and return the dataset with the predictions. These ones are named by "p_" at the beginning of the output name.</p>

```python
x_p = ["x1", "x2"]
data_pred = dataset.loc[:10, x_p]
data_prediction = model.predict(data_pred, x_p)
```

<h2>Efficiency Scores</h2>
<p>The repository has four ways to calculate the efficiency score of EAT model. The first one is the model BCC output oriented. The second one is the model BCC output oriented of CEAT. The third one is the model BBC input oriented. The last one is the model DDF.</p>
<p>To do that, the model EAT of scores is carried out.</p>

```python
mdl_scores = eat.Scores(dataset, x, y, model.tree)
```

<p>Then, the four models exposed before are called.</p>

```python
#Fit BCC output oriented of EAT
mdl_scores.BCC_output_EAT()
#Fit BCC output oriented of CEAT
mdl_scores.BCC_output_CEAT()
#Fit BCC input oriented of EAT
mdl_scores.BCC_input_EAT()
#Fit DDF of EAT
mdl_scores.DDF_EAT()
```

<p>In addition, the model of BCC output oriented of FDH and its DDF model are included.</p>

```python
#Fit BCC output oriented of FDH
mdl_scores.BCC_output_FDH()
#Fit DDF of FDH
mdl_scores.DDF_FDH()
```

<p>Also, this two models are calculated of DEA.</p>

```python
#Fit BCC output oriented of DEA
mdl_scores.BCC_output_DEA()
#Fit DDF of DEA
mdl_scores.DDF_DEA()
```

<h2>Analysis of the PISA data set</h2>
<p>A empirical example of the full operation of the EATpy package is detailed below. The data set used is that of PISA, located in the EATpy/data set folder. It can also be downloaded from https://github.com/MiriamEsteve/EATpy/tree/main/data%20set.</p>

```python
import pandas
import eat
import graphviz

#Read the data set
dataset = pd.read_excel("PISA.xlsx", sheet_name = "data")

#Set the names of the input and output columns
x = ["SCMATEDU", "ESCS", "PRFAL100"]
y = ["PVMATH", "PVREAD, "PVSCIE"]

#Set the stop rule and the number of folds for cross validation
numStop = 5
fold = 5

#Adjust and create the EAT model
model = eat.EAT(dataset, x, y, numStop, fold)
model.fit()


#Drawing the EAT tree
dot_data = model.export_graphviz('EAT')
graph = graphviz.Source(dot_data, filename="tree", format="png")
graph.view()


#Adjust and model the efficiency calculations of the EAT
mdl_scores = eat.Scores(dataset, x, y, model.tree)

#Fit BCC output oriented of EAT
mdl_scores.BCC_output_EAT()
#Fit BCC output oriented of DEAEAT
mdl_scores.BCC_output_DEAEAT()
#Fit BCC input oriented of EAT
mdl_scores.BCC_input_EAT()
#Fit DDF of EAT
mdl_scores.DDF_EAT()

#Fit BCC output oriented of FDH
mdl_scores.BCC_output_FDH()
#Fit DDF of FDH
mdl_scores.DDF_FDH()

#Fit BCC output oriented of DEA
mdl_scores.BCC_output_DEA()
#Fit DDF of DEA
mdl_scores.DDF_DEA()
```
