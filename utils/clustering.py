import numpy as np
import pandas as pd
    
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import scaffoldgraph as sg

def cluster_by_scaffold(bioactivities, descriptors_df, n_clusters=None):
    
    mols = bioactivities['mols'].values
    _ = [mol.SetProp('_Name', d_hash) 
         for mol, d_hash in zip(bioactivities['mols'], 
                                bioactivities['mol_hash'])]
    
    tree = sg.ScaffoldTree.from_supplier(mols, progress=True)
    
    scaffolds_in_tree = [[s for s in tree.get_scaffolds_in_hierarchy(i)]
                         for i in range(tree.min_hierarchy(), 
                                        1+tree.max_hierarchy())]
    
    mols_scaffolds = [tree.get_scaffolds_for_molecule(d_hash) 
                      for d_hash in bioactivities['mol_hash']]
    
    labels = [scaffolds_in_tree[0].index(s[-1]) 
          if (len(s) and (s[-1] in scaffolds_in_tree[0])) 
          else -1 
          for s in mols_scaffolds]
    labels = np.array(labels)
    full_labels = np.array(labels)
        
    # We join small clusters to get total of n_clusters
    if n_clusters:
        
        knn = make_pipeline(StandardScaler(), PCA(8), KNeighborsClassifier())
    
        X = descriptors_df.values
        y_label = labels        
        
        labels_counts = pd.DataFrame(labels).value_counts()
        labels_counts = labels_counts.sort_values(ascending=False)
        labels_counts = labels_counts.iloc[:n_clusters]
        
        mask = np.isin(labels, np.concatenate(labels_counts.index))
        X_train, y_label_train = X[mask], y_label[mask]
        X_test = X[~mask]
        
        labels[~mask] = knn.fit(X_train, y_label_train).predict(X_test)
    
    return labels, full_labels

def plot_tsne(bioactivities, descriptors_df, labels):
    
    tsne = TSNE(2, learning_rate='auto', init='pca')
    std = StandardScaler()
    pca = PCA(8)

    trans = make_pipeline(std, pca, std, tsne)
    
    descriptors = descriptors_df.loc[bioactivities['mol_hash']].values
    descriptors_trans = trans.fit_transform(descriptors)
    
    sns.set('talk')
    plt.figure(figsize=(7, 7))
    plt.title('Top clusters in descriptor space')
    
    mask = (labels!=-1) & (labels!=np.max(labels))
    plt.scatter(*descriptors_trans[mask].T, c=labels[mask], cmap='tab20', s=20)
    
    mask = labels==-1
    plt.scatter(*descriptors_trans[mask].T, c='black', s=20)
        
    plt.show()