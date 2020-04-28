# Drug-Design
Drug interaction using Machine Learning

Developing and discovering drugs is difficult, with clinical trials and approval from authorities and organisations.
One of the best alternative is to use the combination of already existing drugs, i.e. to combine their effects on pathogen. 
However, combination of drugs require huge efforts since any two drugs can have unexpected outcomes on human body. 
Following factors, make drug combination, a difficult and challenging problem-
1. Large space of drugs
2. Dosage of different drugs
3. Chemical properties of drugs
4. Nature of target pathogen

The number of drug combinations is very astronomically high

Machine Learning comes to rescue - 
1. The integration of chemical and genetic information of different drugs can be utilised to predict an interaction score between drugs. 
2. Chemogenomic profiling of drugs help in determining synergy or antagony among two drugs. 
3. Predictive model developed for one pathogen environment can be utilised to predict drug interaction in another pathogen environment    using orthologous genes.

One of the most widely used dataset for the purpose is available from

<b>Nichols et.al., Phenotypic landscape of a bacterial cell., Cell, 144(1), 143-156, Jan 2011</b>

Drug interaction data of 15 drugs in pairwise combination and 72 drugs in pairwise combination is available in E.coli pathogen environment.
It is obtained by comparing the interaction of antibiotic with strain of E.coli, obtained after deleting that gene and that of wild-strain E.coli. The difference in interaction score provides information regarding sensitivity or resistivity between antibiotic and gene.

Data for 324 drug conditions i.e. drugs with different doses and 3979 genes have been obtained. 
Interaction score < -0.5 signifies synergy  while interaction score > 1 signify antagony.

Two different approaches-

1. Since interaction score were provided, it is naturally a Regression task in terms of machine learning.

2. But since synergy and antagony can be defined from the interaction score in terms of range, it can be seen as a potential classification task also. 

We implemented both these approaches, Regression and Classification

 


