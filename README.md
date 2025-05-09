Files and their descriptions:

Data_Manipulation - Files for preprocessing EEG datasets

Preprocessing.ipynb: basic pre-processing. Reads in data from EEG files and does basic preprocessing of the data and stores it within a .npz file
Advanced Preprocessing.ipynb: takes EEG data and performs various DSP techniques to the data, such as applying bandpass filters, computing cross-spectral density, ICA to remove artifacts, and stores train/val/test datasets.

KNN-Based Models - Python Files that were used to train the models based off of Preprocessing.ipynb with KNN algorithm for mapping edges and indexes of the GNN

Train_cnn_encoder.py: Trains the CNN encoder only for classification and saves weights to be used later when merged
Train_gnn_pretrained_cnn.py: Complementary file to the one above. Trains GNN with respect to taking the CNN encoder
Train_cnn_gnn_together.py: To train both cnn encoder and gnn decoder together, run this.

EEG 10-10 Based Models - Python Files that were used to train models based off of Advanced Preprocessing.ipynb with EEG 10-10 system being basis for edges and indexes of GNN
