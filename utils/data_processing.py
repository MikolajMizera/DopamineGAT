import pickle
from hashlib import md5

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from pandarallel import pandarallel
from mordred import Calculator, descriptors

class ChemBLDataset:
    """
    Handles data curation for ChemBL molecues.
    See: Mizera, M., Latek, D., & Cielecka-Piontek, J. (2020). 
    Virtual Screening of C. Sativa Constituents for the Identification of 
    Selective Ligands for Cannabinoid Receptor 2. International journal of 
    molecular sciences, 21(15), 5308.
    for more detials about curation steps.
    """
    
    def __init__(self, sdf_fname, verbose=True):
        
        self.sdf_fname = sdf_fname
        self.verbose = verbose

    def __load_sdf(self):
        
        mols = Chem.SDMolSupplier(self.sdf_fname, removeHs=False)
        mols = [m for m in mols if m]
        
        bioactivities = []
        for mol in mols:
            
            try:
                record = mol.GetPropsAsDict()
            except Exception as e:
                if self.verbose:
                    print(e)
                    print('%s: %s',(self.sdf_fname, 
                                    mol.GetProp('Molecule ChEMBL ID')))
                continue
            
            record['mols'] = mol
            bioactivities.append(record)
    
        bioactivities = pd.DataFrame.from_dict(bioactivities)
        
        return bioactivities

    def get_curated(self, 
                    mw_threshold=600, 
                    confidence_thershold=7,
                    assays_thershold=1,
                    mol_hash='inchikey'):
        
        # Tracks number of removed observations after each curation step
        observation_counts = []
        
        # 1. Load and join metadata
        self.bioactivities = self.__load_sdf()        
        observation_counts.append(len(self.bioactivities))
        
        # 2. Select binding assays only
        mask = self.bioactivities['Assay Type']=='B'
        self.bioactivities = self.bioactivities[mask]
        observation_counts.append(len(self.bioactivities))
        
        # 3. Remove moelcules with MW above threshold
        mask = self.bioactivities['mols'].apply(ExactMolWt)<mw_threshold
        self.bioactivities = self.bioactivities[mask]
        observation_counts.append(len(self.bioactivities))
        
        # 4. Remove records with low confidence
        mask = self.bioactivities['Confidence Score']>confidence_thershold
        self.bioactivities = self.bioactivities[mask]
        observation_counts.append(len(self.bioactivities))
        
        # 5. Remove small assays
        assay_sizes = self.bioactivities['Document ChEMBL ID'].value_counts()
        mask = assay_sizes>assays_thershold
        mask = np.isin(self.bioactivities['Document ChEMBL ID'], 
                       mask[mask].index)
        self.bioactivities = self.bioactivities[mask]
        observation_counts.append(len(self.bioactivities))

        # 6. Remove categorical data
        cat_mask = pd.isna(self.bioactivities['Standard Value'])
        cat_mask |= (self.bioactivities['Standard Relation'] != "'='")
        self.bioactivities = self.bioactivities[~cat_mask]
        observation_counts.append(len(self.bioactivities))
        
        # 7. Transform bioactivity units inplace
        self.__transform_activities_to_pki()
        observation_counts.append(len(self.bioactivities))
        
        # 8. Add inchikey
        inchikeys = self.bioactivities['mols'].apply(Chem.inchi.MolToInchiKey)
        self.bioactivities['inchikey'] = inchikeys
        
        # 9. Calculate descriptors
        self.descriptors_df = self.__calculate_descriptors()
        
        # 10. Remove duplicates with desired hash
        self.__calculate_hash(mol_hash)
        self.__merge_duplicates()
        
        # Remove duplicated descriptor vectors
        unique_hashes = self.bioactivities['mol_hash']
        dup_mask = ~self.descriptors_df.index.duplicated(keep='first')
        self.descriptors_df = self.descriptors_df[dup_mask].loc[unique_hashes]
        
        observation_counts.append(len(self.bioactivities))
        
        self.bioactivities.columns = ['mol_hash', 'mean', 'std', 
                                      'replicates_count', 'replicates', 'mols']
                    
        return self.bioactivities, observation_counts, self.descriptors_df

    def __merge_duplicates(self, allow_std=0.5):
    
        bioactivities_grouped = self.bioactivities.groupby('mol_hash')
        col_aggregators = {'value': ('mean', 'std', 'count', list),
                           'mols': 'first'}
        bioactivities_grouped = bioactivities_grouped.agg(col_aggregators)
        mask = (bioactivities_grouped[('value', 'std')] < allow_std) 
        mask |= pd.isna(bioactivities_grouped[('value', 'std')])
        self.bioactivities = bioactivities_grouped[mask].reset_index()
    
    def __calculate_hash(self, mol_hash):
        
        if mol_hash=='descriptor':
                  
            
            md5_desc = lambda x: md5(pickle.dumps(x)).hexdigest()
            desc_hash = self.descriptors_df.apply(md5_desc, axis=1)
            self.bioactivities['mol_hash'] = desc_hash.values
            self.descriptors_df.index = desc_hash
            
        elif mol_hash=='inchikey':
            self.bioactivities['mol_hash'] = self.bioactivities['inchikey']
            
        else:            
            raise NotImplementedError('Specified hashing is not implemented.')
        
    
    def __calculate_descriptors(self):
        
        pandarallel.initialize(progress_bar=True)
        
        calc = Calculator(descriptors, ignore_3D=False)
    
        mol_descriptors = self.bioactivities.groupby('inchikey').first()[
            'mols'].parallel_apply(lambda x: calc(x))
        indices = mol_descriptors.index
        mol_descriptors = np.vstack(mol_descriptors).astype(float)
        mol_descriptors = (mol_descriptors-mol_descriptors.mean(0)
                           )/mol_descriptors.std(0)
        mol_descriptors = mol_descriptors[:, ~np.any(np.isnan(mol_descriptors), 
                                                     0)]
    
        descriptors_df = pd.DataFrame(index=indices, data=mol_descriptors)
        descriptors_df = descriptors_df.loc[self.bioactivities['inchikey']]
    
        pandarallel.initialize(progress_bar=False)

        return descriptors_df
    
    def __transform_activities_to_pki(self):
    
        n_log_transform = ['Ki'] # 'IC50'
        no_transform = ['pKi'] # 'pIC50'
        log_inverse_unlog = []
    
        types_allowed = n_log_transform + log_inverse_unlog + no_transform
        self.bioactivities = self.bioactivities[np.isin(
            self.bioactivities['Standard Type'], types_allowed)]
    
        self.bioactivities['value'] = np.nan
    
        # negative log transform
        mask = np.isin(self.bioactivities['Standard Type'], n_log_transform)
        raw_values = self.bioactivities[mask]['Standard Value']
        converted = -np.log10(10**-9 * raw_values)
        self.bioactivities.loc[mask, 'value'] = converted
    
        # log inverse unlog transform
        mask = np.isin(self.bioactivities['Standard Type'], log_inverse_unlog)
        raw_values = self.bioactivities[mask]['Standard Value']
        converted = -np.log10((10**-9) * (1 / (10**raw_values)))
        self.bioactivities.loc[mask, 'value'] = converted
    
        # no transform
        mask = np.isin(self.bioactivities['Standard Type'], no_transform)
        raw_values = self.bioactivities[mask]['Standard Value']        
        self.bioactivities.loc[mask, 'value'] = raw_values
        
            