#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:42:08 2020

@author: backpropagator
"""
import pandas as pd
import numpy as np
from scipy.io import loadmat

z = 2 # for handling chemogenomics data
interaction_filename = './training_set.xlsx'
annotation_filename = './matching.xlsx'
chemogenomics_filename = './drug_gene_data.xlsx'
test_interaction_filename = './validation_set.xlsx'

interaction_file = pd.read_excel(interaction_filename)

interaction_pairs = np.array(interaction_file.iloc[:,:2]) #storing interaction pair
interaction_score = np.array(interaction_file.iloc[:,2]) #storing scores of particular interaction pair
drugs_all = np.unique(interaction_pairs) # finding unique drugs present in list of interaction pairs

annotations = pd.read_excel(annotation_filename)
drug_id = np.array(annotations.iloc[:,0]) #drug id, used in interaction data
chemgen_id = np.array(annotations.iloc[:,1]) # chemgen id, used in chemogenomics data

drugnames_all = np.copy(drugs_all) #copying, used to have two different arrays with unique entries one with abbr and one with detailed

for i in range(drug_id.shape[0]):
    drugnames_all[drugnames_all==drug_id[i]] = chemgen_id[i]

drugpairsname_cell = np.copy(interaction_pairs)

for i in range(drug_id.shape[0]):
    drugpairsname_cell[drugpairsname_cell==drug_id[i]] = chemgen_id[i]

chemogenomics_file = pd.read_excel(chemogenomics_filename) # reading chemogenomics data file
    
phenotype_num = np.array(chemogenomics_file.iloc[1:,1:]) #actual data
probelist = np.array(chemogenomics_file.iloc[1:,0]) # list of genes
conditions = np.array(chemogenomics_file.iloc[0,1:]) #list of antibiotics

plist = np.empty_like(probelist)
    
for i in range(probelist.shape[0]):
    tem = probelist[i].split('-')
    try:
        plist[i] = tem[1]
    except:
        plist[i] = tem[0]

pnum_array = loadmat('./quantilenorm.mat') #importing data from matlab after processing
phenotype_num = np.array(pnum_array['pnum'])

cell_z1_list = [] #to capture the list of genes which are more sensitive
lte = [] #to capture list of such genes for every particular condition
list_phenotype = [] #to find the unique list of such genes

for i in range(phenotype_num.shape[-1]):
    te = plist[phenotype_num[:,i]<-1*z] # for particular condition, finding the most important genes
    cell_z1_list.append(te) # appending list of such genes in list
    lte.append(len(te)) # keeping in account the number of such genes for particular condition
    for j in range(len(te)): 
        list_phenotype.append(te[j]) # for finding the number of such genes in the whole dataset
        
np_array = np.array(list_phenotype)
phenotype_labels = np.unique(np_array) #list of genes in whole dataset

nichols_t = np.zeros((phenotype_labels.shape[0],phenotype_num.shape[-1])) #storing binary dataset

for i in range(nichols_t.shape[0]):
    for j in range(nichols_t.shape[-1]):
        for k in range(len(cell_z1_list[j])):    
            if phenotype_labels[i] == cell_z1_list[j][k]:
                nichols_t[i,j] = 1
                
phenotype_data = np.copy(nichols_t)

ix = np.zeros(drugpairsname_cell.shape) # if drug at particular position is present in our list of data, it turns the value here to 1 else 0

for i in range(drugpairsname_cell.shape[0]):
    for j in range(drugpairsname_cell.shape[1]):
        if drugpairsname_cell[i,j] in conditions:
            ix[i,j] = 1
    
ix = ix.astype(np.uint8)
ix = np.where(ix[:,0]*ix[:,1]) #finding index in drug pairs cell where both drugs have data available

train_interactions = drugpairsname_cell[ix] #list of pairs whose data is available and hence can be used for training the model
    
trainxnscores = interaction_score[ix] #interaction score corresponding to interaction pairs

traindrugs = np.unique(train_interactions) #finding unique drugs in whole training set

pos = np.zeros(traindrugs.shape) # with the idea to keep the indexes where data of particular drug of training set lies in phenotype data

for i in range(pos.shape[0]):
    pos[i] = np.where(conditions == traindrugs[i])[0][0]
    
pos = pos.astype(np.uint16)

trainchemgen = phenotype_data[:,pos] #dataset of the whole phenotype relevant to training set
trainchemgen = trainchemgen.astype(np.uint8)

chemgen = np.copy(trainchemgen) #copy to avoid any change in real data
alldrugs = np.copy(traindrugs) 
interactions = np.copy(train_interactions)

xtrain = []
ytrain = []
for i in range(interactions.shape[0]):
    ix1 = []
    list_drugs = interactions[i]
    for d in list_drugs:
        if d in alldrugs:
            index = np.where(alldrugs==d)[0][0]
            ix1.append(index)

    t1 = chemgen[:,ix1[0]]
    t2 = chemgen[:,ix1[1]]
    sigma = t1 + t2
    delta = t1 + t2
    delta[delta==2] = 0
    t3 = np.concatenate((sigma,delta),axis=0)
    
    xtrain.append(t3)
    ytrain.append(trainxnscores[i])

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

np.save('./data/xtrain.npy',xtrain)
np.save('./data/ytrain.npy',ytrain)

# generating test interaction dataset
test_interaction_file = pd.read_excel(test_interaction_filename) #loading test interaction data
test_interaction_pairs = np.array(test_interaction_file.iloc[:,:2])
test_interaction_score = np.array(test_interaction_file.iloc[:,2])

test_drugs_all = np.unique(test_interaction_pairs) #finding unique drugs in test interaction data

test_drugnames_all = np.copy(test_drugs_all) 

for i in range(drug_id.shape[0]):
    test_drugnames_all[test_drugnames_all==drug_id[i]] = chemgen_id[i] #replacing unique drug names with the one in the official chemogenomics data

test_drugpairnames_all = np.copy(test_interaction_pairs)

for i in range(drug_id.shape[0]):
    test_drugpairnames_all[test_drugpairnames_all==drug_id[i]] = chemgen_id[i] #replacing drug interaction pair names with official chemogenomics data
    
test_ix = np.zeros(test_drugpairnames_all.shape) #finding which test interactions pairs have their complete data in chemogenomics profile

for i in range(test_drugpairnames_all.shape[0]):
    for j in range(test_drugpairnames_all.shape[1]):
        if test_drugpairnames_all[i,j] in conditions:
            test_ix[i,j] = 1

test_ix = test_ix.astype(np.uint8)
test_ix = np.where(test_ix[:,0]*test_ix[:,1])

test_interactions = test_drugpairnames_all[test_ix] #list of test interaction pairs whose complete data is available
    
testxnscores = test_interaction_score[test_ix] #list of interaction scores

testdrugs = np.unique(test_interactions) # list of unique drugs present in test dataset

test_pos = np.zeros(testdrugs.shape) #finding index where data of that drug is located

for i in range(test_pos.shape[0]):
    test_pos[i] = np.where(conditions == testdrugs[i])[0][0]

test_pos = test_pos.astype(np.uint16)

testchemgen = phenotype_data[:,test_pos]
testchemgen = testchemgen.astype(np.uint8)

chemgen = np.copy(testchemgen) #copy to avoid further misuse in original data
alldrugs = np.copy(testdrugs)
interactions = np.copy(test_interactions)

xtest = []
ytest = []

for i in range(interactions.shape[0]):
    ix1 = []
    list_drugs = interactions[i]
    for d in list_drugs:
        if d in alldrugs:
            index = np.where(alldrugs==d)[0][0]
            ix1.append(index)

    t1 = chemgen[:,ix1[0]]
    t2 = chemgen[:,ix1[1]]
    sigma = t1 + t2
    delta = t1 + t2
    delta[delta==2] = 0
    t3 = np.concatenate((sigma,delta),axis=0)
    
    xtest.append(t3)
    ytest.append(testxnscores[i])

xtest = np.array(xtest)
ytest = np.array(ytest)

np.save('./data/xtest.npy',xtest)
np.save('./data/ytest.npy',ytest)





  



