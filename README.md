# ml_projects


## Purpose
The overall aim of the project is to analyse the possibility of heart attack based on the chosen Heart  Disease  Data  Set  obtained  by  Cleveland  Clinic Foundation.
(source: https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility?fbclid=IwAR1GzOnSDjH10OLucbRHZrvSSesKWnK1IkdzEboQQg-gSXMy-SKtnxzD5j4 )
The  observations  consist  of  an  individual  patients  and  their  medical  records  as  attributes.   The attributes used for analysis are limited to the recommended 14 attributes by the source of the data set andthere are together 303 observations.  The dimensions of the data set are therefore 303x14 (NxM).

## Part I: :

The objective of this report is to apply methods regarding: processing of data, feature extraction, PCA and Data visualization, in order to get a basic understanding of the data prior to the further analysis which will follow in Part II. 

The variable one would like to predict using this data set is the chance of heart attack possibility (target) for each patient based on their medical records. Although the data is mixed, presenting a combination of both discrete and continuous variables, it has no missing values, and the observation entries seem to be structured. 

### main.py
Can be ignored. Used for references.

### data_aquisition.py
Presents the data processing according to information found in data_visualization.py. 
* X_wo: datamatrix inclusing all 14 attributes.
* X: X_wo having outliers removed. Selection has been made based on information gathered from boxplots in data_visualization.py
* X_sel_wo: subset of data selected for the scope of this project. The selection includes all the continous attributes (age,trestbps, chol, thalach and oldpeak) and some discrete variables, which from previous analysis prove to be somewhat significant to the dataset (sex, cp, slope and thal). 
* X_sel: X_sel_wo having outliers removed. Data visualization is mostly based on X_sel and X_cont.
* X_cont: Subset including continous attributes. Since they are on a different scale, the continous attributes have been standardized in data_visualization.py (X_cont_stand)

### data_visualization.py
Includes:
* histograms: for all variables, for continuous variables and matching pdf from N-distribution;
* boxplots: for all variables, for continuous standardized variables, for all variables - where continuous var. are standardized
* correlation graphs: where each of the two selected attributes are plotted against each other
* correlation matrix in Latex tabular form (check console after running the program)
* 3D plot of age, trestbps and chol
* data matrix

Data visualization has helped with the selection process and observing more information about our data. Using boxplots, outliershave been identified and removed, as well as the column thal which presented some odd measurements. Some information about the spread of the data based on each class has also been gained. Histograms revealed that some continuous attributes (age, trestbps, chol and thalach) seem to follow a normal distribution. Continuous attributes have been plotted against each other. Observing the correlation values some correlation has been observed between age and the maximum heart rate achieved (thalach) of a patient, as well as between maximum heart rate achieved and the depression of the ST segment (oldpeak) of their electrodiagram results.

### pca.py
Includes:
* Graph which explains variance by principal components
* Data observations ploted on PC1 and PC2 plane
* PCA component coefficients

The principal component analysis helped with analysing of which the continuous attributes contribute the most to the data variation. The results also show that if the focus would be only on the 90% of the variance the set of continuous attributes of size 5 can be shrunk into a set of 4 principal components, hence limiting the amount of attributes. Lastly the correlation of the two most significant principal components showed a slight tendency of the pc1 component to predict the target however the definite result is not clear.

### Conclusion for Part I:

Overall the the primary machine learning aim appears to be feasible. The data selected might be suitable for applying a classification model - this is also supported by previous analysis of the data. One could thus predict the heart attack possibility based on those attributes, although the validation of the model is
yet to be discussed in Part II.


