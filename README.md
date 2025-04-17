# **Introduction**
In this JupyterNoteBook, we applied several programming skills to study a biological dataset. We performed some statistical analysis on the dataset. Then, we used a machine learning supervised approach named Gaussian Naive-Bayes classifier, intending to classify compounds as active or inactive.
The dataset was from a previous publication (https://pubs.acs.org/doi/10.1021/ja512751q) in which the authors reported experimental and computational (using the FEP method) binding free energies (deltaG) for different molecular targets interacting with several bioactive compounds. The deltaG values from the experiment and computational method were compared using regression analysis. The variance and covariance were calculated for the whole data in two sets of computational calculations.
In the final step, the dataset is complemented with additional features (columns) corresponding to physicochemical descriptors obtained from chemical structures (in SMILE format) using RDkit software.  The calculated descriptors were MW, LogP, TPSA, HBD, HBA, volume, numAtoms, RotatableBonds, and Lipinskiviolations.  Those descriptors are Lipinski rules and allow for predicting drug compounds' oral bioavailability. 
Using those additional nine features, we implemented a classification model known as Gaussian Naive Bayes (NB), which allowed us to train the NB model to classify active and inactive compounds. The model was tested using 30% of the dataset and gave metrics like accuracy and F1-score that suggest that it has good predictive capabilities for this dataset.

# **Dataset and Python scripts:**
Designing tight-binding ligands is a primary objective of small-molecule drug discovery. Over the past few decades, free-energy calculations have benefited from improved force fields and sampling algorithms and the advent of low-cost parallel computing. However, achieving the accuracy level needed to guide lead optimization (∼5× in binding affinity) for a wide range of ligands and protein targets has proven challenging. In the chosen dataset, Wang *et al.* (J. Am. Chem. Soc. 2015, 137, 7, 2695–2703) reported the computational and experimental binding free energy values for 199 compounds grouped according to the following molecular targets (proteins) they interact with:
|Protein | # compounds|
|--------|------------|
|BACE |36 |
|p38a  | 34|
|CDK2 | 16 |
|JNK1 |21  |
| MCL1 | 42 |
|Thrombin |11 |
| Tyk2  | 16 |
| PTP1B | 23 |

The authors kindly provided the dataset and chemical structures of compounds and proteins.  The compounds were shared as .sdf files, and we transformed them into SMILES files using the software Openbabel (https://openbabel.org/). Those SMILES files codifying the compound's chemical structure were used to calculate the nine physicochemical descriptors used as additional features in the classification machine learning model.  The Python script (rdkit_descriptors.py) was used to obtain the nine additional features using the software RDKit (https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors). 
The classification machine learning model used those physicochemical features as target features (in the X-axis). They were Molecular weight (MW), Volume, LogP (Hydrophobicity), HBD (Hydrogen bond donors), HBA (Hydrogen bond acceptors), natoms (number of heavy atoms), nviolations (number of Lipinski rule violations), rotatablebonds (number of rotable bonds), and TPSA.
Once descriptors for all ligands were calculated and saved as CSV files, they were merged into a unique CSV file for further statistical and supervised machine learning analysis application using the Python script merging_csv-files.py. 
The final step in completing the dataset was to map and merge the CSV files for physicochemical descriptors with the CSV file with biological data from experimental and computational calculations. The objective was to get the columns "Exp. dG" and "Pred. dG" from the publication dataset and map the name of every ligand to assign that information to the dataset containing the physicochemical descriptors calculated with RDkit. For this purpose, the Python script mapping-merging_data.py was used to obtain the final dataset with features (compounds physicochemical descriptors) and target features like biological activity (Exp. dG).

# **Sprint1-2: Regression model between experimental (Exp. dG) and computational (Pred. dG) binding free energies**

In the first sprint, we selected the dataset according to our interests in any area of expertise. During the second sprint, we performed some exploratory data analysis (EDA), like calculating variance and covariance on the dataset.  Furthermore, we applied a simple regression model to two variables reported in the dataset: experimental binding free energy ("Exp. dG" column) and computational binding free energy ("Pred. dG" column). 

The former relates to measuring compound affinity by its molecular target using experimental protocols like thermal titration calorimetry or biochemical approaches. The affinity expressed in kcal/mol can be obtained directly from experimental design or can be transformed from IC50 values to binding free energy values using the equation dG = - RT ln IC50, where dG is binging free energy (kcal/mol), R is the gas constant, T is temperature, and IC50 is the inhibitory concentration (commonly in nM) required to inhibit the enzyme at 50% of its biological activity. 

The second variable (Pred. dG) relates to a computational prediction using molecular dynamics simulations and the free energy perturbation (FEP) approach to estimate the binding free energy from simulations. The computational binding free energy is commonly reported in kcal/mol, and it is expected to correlate with its experimental counterpart to get a predictive model in which, from computational modeling and calculations, it is possible to infer if a new compound might have or not affinity by its molecular target when tested in the wet laboratory.

Our regression model results suggest a strong correlation between two variables, "Exp. dG" and "Pred. dG," with a **Pearson correlation coefficient** of **0.82**.
This roughly means that "Pred. dG" values calculated computationally can explain 82% of the variability of "Exp. dG". Moreover, this result suggests that the regression model can be used to predict the experimental affinity of a new compound using the computational protocol that estimates binding free energies with the FEP approach (Pred. dG).

![image](https://github.com/user-attachments/assets/14b99b43-4eee-4853-80c3-da948ae70b1b)

The results for variance (Exp. dG and Pred. dG) and covariance (Exp. dG-Pred. dG) indicate that data points in "Exp. dG" are less spread, **1.78**, than data points in the "Pred. dG" variable, which shows a variance of **2.60**.  The plots reported in Jupiter notebook allow us to observe this spreading of data points clearly when the Mean is taken as a reference point in the plot (see Figures 2 and 3 in Jupiter notebook). Finally, the covariance measures the relationship between X and Z. It denotes the mean for each variable by X and Z. As with the correlation coefficient, positive values indicate a positive relationship and negative values indicate a negative relationship. In this study case, the covariance gave a positive value of 1.76, suggesting that the relationship among the analyzed variables (Exp. dG and Pred. dG) is positive or, in other words, that both variables change in the same direction with a positive slope as indicated in the regression plot.

# **Supervised machine learning model: Gaussian Naive-Bayes classifier**

In the third sprint, the task assigned was to build a supervised machine-learning model on the dataset selected. The chosen model was the Gaussian Naive-Bayes (NB) classifier that helped us to train and test our dataset to predict active and inactive compounds using the physicochemical features (in the X-axis) and the "Exp. dG" as the target feature (in the Y-axis). Some additional preparation of the dataset was necessary before implementing the NB classifier:

#### 1. The target feature "Exp. dG" was converted into a binary variable where 1 means active and 0 means inactive. To do so, we established a threshold for "Exp. dG" values, where values below -7.0 kcal/mol were considered active (value 1); otherwise, the compound is inactive (value 0). 
#### 2. Some categorical and numerical features (columns) were dropped from the dataset because the Gaussian NB model deals only with numerical values.  For instance, the features "SMILES", "ID", and "Pred. dG" did not take part in this analysis.
#### 3. We checked the dataset to find missing data points.
#### 4. The features were assigned to the X and Y axes. The features MolWt,	LogP,	HBD,	HBA,	RotatableBonds,	TPSA,	Volume,	LipinskiViolations, and	NumAtoms were assigned to the X-axis, and the target feature "Exp. dG" was assigned to the Y-axis.

Then, the data set was split into training (70%) and test (30%) datasets, and the Gaussian NB classifier was applied to the training dataset using the scikit-learn module "sklearn.naive_bayes." After the training cycle, the test dataset is challenged by predicting the active and inactive compounds using the trained Gaussian NB classifier model.
Finally, some metrics (accuracy, F1-score, and confusion matrix) are calculated to assess the classifier model implemented with the Gaussian NB algorithm.

The results suggest that the Gaussian NB classifier is accurate 78% and has an F1 score of 72%. The confusion matrix can give a clearer picture of the predicted categories achieved by the Gaussian NB classifier. For instance, the number of TruePositives (TP), TrueNegatives (TN), FalsePositives (FP), and FalseNegatives (FN) can be extracted from the matrix as shown in Figure 4 in Jupyter Notebook.
From 64 compounds in the test dataset, 49 are predicted as TP, 12 as TN, 2 as FP, and 1 as FN.

![image](https://github.com/user-attachments/assets/a1ea216f-8218-4d6a-9070-0885bb15b63f)


# **Conclusions and Perspectives**

In this term project, we could use the power of Python scripting to implement statistical and supervised machine learning models on a chosen dataset. 
In our case, we could study a biological dataset comprising several compounds with bioactivity for eight proteins or molecular targets. We developed a regression model with a good Pearson coefficient correlation to compare the experimental and computational binding free energies.
In a more challenging exercise, the dataset was complemented with several physicochemical features to implement a supervised machine learning model named Gaussian Naive Bayes classifier. The dataset was split into training (70%) and testing (30%) groups, and the classifier was implemented using sci-kit-learn modules. 
The Gaussian NB classification results suggest that the model is capable of classifying the compound from the test set into active and inactive with an accuracy of 78% and an F1-score of 72%.
The classifier will be tested using several iterations with a random number and different proportions for training and test sets, and the metrics of the Gaussian NB model will be compared against another classifier available in the literature.



 





