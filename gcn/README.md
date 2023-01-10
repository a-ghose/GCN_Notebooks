# Graph Convolutional Network for AD Classification - 2020

Deep Graph Library (DGL) packages were used to make various Graph Convolutional Networks (GCN's) to classify the stages of neurodegenerative diseases. These stages are Subjective Memory Impairment (SMI), Mild Cognitive Impairment (MCI), and finally Alzheimer's Disease (AD).

From DGL, GCN Layers, MoNet, and Chebyshev code was used. Data consisted of white matter connectome data and brain morphometry data.
# Datasets-
* ADNI-201 total patients(179 w/Connectome Data and Morphometry Data)
* IDP-211 total patients(208 w/Connectome Data and Morphometry Data)

# Running Code

## Packages required-
* Deep Graph Library (DGL) - Graph Construction, GCN Layers
* Pytorch
* SciKit Learn
* XGBoost

## Running Location-
* All files were run while mounted in 'src'


# Multimodal Node Classification GCN-
## Args Setup-
* Set lr, epochs, weight decay, folds, dropout

## GCN Class Definition-
* Imported GCNLayer from DGL package.
* Set Net architecture

## Supporting Functions-
* Evaluate: Returns accuracy of model given mask
* Grid Graph: Constructs Node Classification Graph using gaussian metric

## Data-
* Load connectome data from two datasets (164 Regions of Interest (aparc); 84 Regions of Interest(aparc2))
* Each set has connectivity matrix of count and length
* Count: Number of connections at each node
* Length: Length of connections
* All data is appended into one set and Labels are loaded
* Using utils.featsel() , feature selection of 100 most influential (correlative) features were selected. (179 patients, 100 features)
* Morphometry (Structural Brain measurements) data loaded (179 patients, 100 features). 
* Both data combined into final dataset. (179 patients, 200 features).

## Divide Data-
* Slices/LOC of SMI, MCI, and AD are made. (Index for each type - 1,2,3).
* AD (label = 1) v MCI (label=0) feature matrix and label vector is made. (108 patients, 200 features).
* First 100 patients are used to have more equal number of AD and MCI, and for the graph (requires square number of nodes).
* StratifiedKFold is used to collect bin indexes for each fold and to make train and test data sets with the ten total folds. (Indexes stored in bin_ixs)
* First fold index (size 10) used for validation. Next 2 index sets used for testing. Remaining 7 sets used for training. Each set (size 10).

## Graph Construction-
* Create graph for node classification using grid graph functions. (100 nodes).
* Graphs converted to DGL graphs for the model
* Features, labels, graph, device set to gpu or cpu

## Run GCN model (AD v MCI)-
Our model will assign each node in the constructed graph a patient with 200 features. Using the training labels, some of the nodes will be labeled while the ones in the testing set are unlabeled. The model will be trained first on the labeled nodes (training set) Of the train indexes (7 folds of 10 indexes) one will be used for training accuracy and the other 6 will be used for training. These sets are cycled such that each fold is used as for training accuracy at one point. (7 total folds for training) Loss is calculated for all sets. The model is then tested on the validation set/fold. After training, the model is tested by classifying the unlabeled nodes using the evaluate function. Test accuracy and loss are returned. Our model classifies each node as either MCI (0) or AD (1).

## Baseline-
* XGBoost Random Forest Classifier is used to calculate baseline model accuracy.



