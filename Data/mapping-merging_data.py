import pandas as pd

# Load and clean column names
data1 = pd.read_csv('dG-Tabla1.csv', delimiter=';')
data1.columns = data1.columns.str.strip()

data2 = pd.read_csv('kinase-ligands_merged_output.csv', delimiter=';')
data2.columns = data2.columns.str.strip()

# Merge on Ligand (from data1) and ID (from data2)
merged = data2.merge(data1[['Ligand', 'Exp. dG', 'Pred. dG']], left_on='ID', right_on='Ligand', how='left')

# Optional: drop redundant 'Ligand' column (since we have 'ID' already)
merged = merged.drop(columns=['Ligand'])

# Save result
merged.to_csv('../FEP-data.csv', index=False)



