# predicting_enthalpy_machine_learning
This repository contains the work for predicting the formation enthalpy of hydrocarbon radicals adsorbed on Pt(111) using molecular fingerprints and machine learning.  
The developed machine learning models enable the estimation of thermochemical properties for the hydrocarbon surface intermediates much faster than using traditional quantum chemical methods such as density functional theory (DFT) calculation. It eventually leads to help identify which species and structures are low energy and therefore more likely in complex reaction pathways with affordable computational cost.  

## Dependencies
Python (>= 3.xx), NumPy, Pandas, scikit-learn, [xgboost](https://xgboost.readthedocs.io/en/stable/python/)


## Data sets
The data sets include the following information on 384 C2-C6 hydrocarbon species adsorbed on Pt(111) (all relaxed structures).
1) SMILES representation of free radicals in the gas phase  
2) DFT-derived formation enthalpy per carbon  
3) Four molecular fingerprints (GA, GASS, FMF, SVCF)  

## File descriptions
- Four data set files (**GA.csv, GASS.csv, FMF.csv, SVCF.csv**) contain information described in the previous section.  
Please refer to this (a link to be updated upon acceptance of the manuscript) for a detailed description of the four manuscripts used in this study.  
- **Total_3115_GA.csv** includes GA fingerprints for 3115 acyclic C2-C6 hydrocarbon species along with their SMILES notation. This information is used to predict their formation enthalpies using a developed model.
- **ml_model** folder consists of three files. **feature_eng.py** and **ML_regression.py** are modules for feature engineering and machine learning models, respectively. **ml_enthalpy_prediction.ipynb** represents a sample script to train and test the machine learning models.
- **Plots** folder contains 30 parity plots for trained machine learning models, including individual and ensemble models.    

## Results
The manuscript for this study, "Predicting Enthalpy of Hydrocarbon Radicals Adsorbed on Pt(111) Using Molecular Fingerprints and Machine Learning", has been submitted (As of 12/26/2023).  
Main findings and detailed discussion can be found here (a link to be updated upon acceptance of the manuscript)

## Acknowledgments
I thank Charanyadevi Ramasamy, Daniel E. Raser, and Gustavo L. Barbosa Couto for contributing to the DFT calculation. Also, Prof. David Hibbitts provided a valuable discussion on this work in addition to DFT data performed with his student Lydia Thies. Above all, I appreciate Prof. Fuat Celik's supervision of this work.
This work was supported by the National Science Foundation (CBET-1705746). This work used the Extreme Science and Engineering Discovery Environment (XSEDE) located at Pittsburgh Supercomputing Center (PSC), which is supported by National Science Foundation grant number ACI-1548562 (allocation no. CTS200045), the Rutgers Office of Advanced Research Computing, which is supported by Rutgers and the State of New Jersey, and the Rutgers School of Engineering High-Performance Computing for computational resources.




