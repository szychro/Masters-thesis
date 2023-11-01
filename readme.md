# SWAGDPU
A computational data analysis
workflow for patient classification and
synthetic oversampling - from data to dashboards.
![](/Users/szymon/Desktop/Praca/My paper/Images/GIF-SWAGDPU.gif)
---
### Table of contents
* [General info](#general-info)
* [Technical specification](#technical-specification)
* [Usage](#usage)
  * [Data preprocessing](#data-preprocessing)
  * [UMAP](#UMAP)
  * [Feature selection](#feature-selection)
  * [ProWRAS](#ProWRAS)
  * [Classification evaluation](#classification-evaluation)
  * [Dash](#dash)
---
### General info
SWAGDPU is an overall data analysis workflow,
including data integration, developing a
diverse set of machine learning models, and setting up a Dash application.\
This workflow consists of six modules. Each of this tool is very 
different,which results in a 
very versatile pipeline, that could be easily reused for other
datasets and applications on different domains based on the 
individual purposes. All the methods proposed here can be easily
replaced with other algorithms.
---
### Technical specification
| Library    | Version |
|------------|---------|
| pandas     | 2.1.1   |
| numpy      | 1.21.1  |
| matplotlib | 3.3.2   |
| sklearn    | 2.0.2   |
| umap       | 0.1.1   |
| scipy      | 1.11.3  |
| dash       | 2.13.0  |
---
### Usage
#### Data preprocessing
In this module we are preparing the data for the next steps.
In *columns_to_drop* write down the columns to remove. In
*categorical_mappings* set the encoding for the variables.
```python
input_filename = 'path_to_input_file.csv'
output_filename = 'path_to_output_file.csv'
columns_to_drop = ['Sex male', 'sex female', ...] 
categorical_mappings = {'SEX': {"f": 1, "m": 0}, 'CEBPASTAT': {"WT": 1, ...}}
```
In *columns_to_fill_median* write down the continuous variables.
```python
def preprocess_data(filename, columns_to_drop, categorical_mappings=None):
    columns_to_fill_median = ['AGE', 'WBC', 'HB', 'PLT', 'LDH', 'BMB', 'PBB', 'FLT3R']
```
#### UMAP
The file saved in the first module insert here. In this part
you get the view of the UMAP visualization for your data. You
can set the number of clusters and parameters of UMAP. The
column with cluster's labels will be added to the dataset.

#### Feature selection
Use the saved file from the UMAP section. Set the values for
X and y and run the code to get the plots.

#### ProWRAS
Download the[ ProWRAS library ](https://github.com/COSPOV/ProWRAS)
and use the file from the UMAP module. Working with ProWRAS 
requires to use numpy package instead of pandas. To create
the synthetic samples, choose the minor cluster and set the
label to 1, while others to 0.
Set the features, labels and the parameters of ProWRAS. 
```python
synth = Library.prowras.ProWRAS_gen(features, labels,
                                        max_levels=5,
                                        convex_nbd=5,
                                        n_neighbors=5,
                                        max_concov=5,
                                        num_samples_to_generate=1000,
                                        theta=1.1,
                                        shadow=7,
                                        sigma=0.4,
                                        n_jobs=1)
```
#### Classification evaluation
Use the file from the UMAP section or from ProWRAS.
Same as in ProWRAS,
set the label of minor cluster to 1, while others to 0. Here
you can compare the results of the data with and without 
the synthetic samples.
#### Dash
Use the file from the data preprocessing. The file is showing
a code that was used to create the web interface. It allows
to work on the variables and add the new values in real time.



