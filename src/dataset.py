import pickle
import os
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from rdkit.Chem.Lipinski import NumAromaticCarbocycles, NumAromaticRings, NumAliphaticRings, NumAromaticHeterocycles, \
NumHeteroatoms, NumSaturatedHeterocycles, NumSaturatedCarbocycles, NumSaturatedRings, NOCount


def calculate_physicochemical_properties(smiles):
    properties = []
    
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    properties.append([CalcNumRings(x) for x in mols])
    properties.append([NumAromaticCarbocycles(x) for x in mols])
    properties.append([NumAromaticRings(x) for x in mols])
    properties.append([NumAliphaticRings(x) for x in mols])
    properties.append([NumAromaticHeterocycles(x) for x in mols])
    properties.append([NumHeteroatoms(x) for x in mols])
    properties.append([NumSaturatedHeterocycles(x) for x in mols])
    properties.append([NumSaturatedCarbocycles(x) for x in mols])
    properties.append([NumSaturatedRings(x) for x in mols])
    properties.append([NOCount(x) for x in mols])
    properties = np.stack(properties)
    
    properties_dict = {smiles[idx]: properties[idx, :] for idx in range(len(smiles))}
    return properties_dict
    

def extend_sequence(sequence, min_index, max_index, max_length):
    subsequence = sequence[min_index:max_index+1]
    
    current_length = len(subsequence)
    additional_length = max_length - current_length

    left_extension = min(min_index, additional_length // 2)
    right_extension = min(len(sequence) - max_index - 1, additional_length - left_extension)

    extended_min_index = min_index - left_extension
    extended_max_index = max_index + right_extension
    extended_subsequence = sequence[extended_min_index:extended_max_index+1]

    padding_needed = max_length - len(extended_subsequence)
    extended_min_index = extended_min_index - min(extended_min_index, padding_needed)
    
    extended_subsequence = sequence[extended_min_index:extended_max_index+1]
    padding_needed = max_length - len(extended_subsequence)
    return extended_min_index, extended_max_index, padding_needed


def process_interface(datapath, maxlen=800):
    """
    Preprocess the interface regions in protein sequences to extend and pad them to the specified maximum length.
    
    Args:
        datapath (str): path to the preprocessed interface dataframe(csv), which should 
                        contain the minimum index and maximum index of interface residues.
        maxlen (int): maximum length of sequences. default is 800.
    
    Returns:
        pd.DataFrame: dataframe with additional columns for adjusted beginning(beg_idx), end(end_idx) and padding length(pad_idx).
    """
    
    df = pd.read_csv(datapath)

    if 'min_index1' in df.columns:
        for i in range(len(df)):
            sequence = df.loc[i, 'uniprot_sequence1']
            min_index = int(df.loc[i, 'min_index1'])
            max_index = int(df.loc[i, 'max_index1'])
            extended_min_index, extended_max_index, padding_needed = extend_sequence(sequence, min_index, max_index, maxlen)

            df.loc[i, 'beg_idx1'] = int(extended_min_index)
            df.loc[i, 'end_idx1'] = int(extended_max_index)
            df.loc[i, 'pad_idx1'] = int(padding_needed)

            sequence = df.loc[i, 'uniprot_sequence2']
            min_index = int(df.loc[i, 'min_index2'])
            max_index = int(df.loc[i, 'max_index2'])
            extended_min_index, extended_max_index, padding_needed = extend_sequence(sequence, min_index, max_index, maxlen)

            df.loc[i, 'beg_idx2'] = int(extended_min_index)
            df.loc[i, 'end_idx2'] = int(extended_max_index)
            df.loc[i, 'pad_idx2'] = int(padding_needed)

    else:
        df['beg_idx1'] = 0
        df['end_idx1'] = df['uniprot_sequence1'].apply(lambda x: len(x) if len(x) <= maxlen else maxlen)
        df['pad_idx1'] = maxlen - df['end_idx1']
        df['beg_idx2'] = 0
        df['end_idx2'] = df['uniprot_sequence2'].apply(lambda x: len(x) if len(x) <= maxlen else maxlen)
        df['pad_idx2'] = maxlen - df['end_idx2']
    return df



class PPIInhibitorDataset():
    def __init__(self, datapath, device, maxlen=800):
        self.device = device
        
        df = pd.read_csv(datapath).drop(columns=['uniprot_id1', 'uniprot_id2'])
        interface = pd.read_csv('data/processed_interface.csv')
        df = pd.merge(df, interface, left_on='ppi_label', right_on='ppi_label')
        
        self.beg_idx1 = df['beg_idx1'].astype(int).tolist()
        self.end_idx1 = df['end_idx1'].astype(int).tolist()
        self.pad_idx1 = df['pad_idx1'].astype(int).tolist()
        self.beg_idx2 = df['beg_idx2'].astype(int).tolist()
        self.end_idx2 = df['end_idx2'].astype(int).tolist()
        self.pad_idx2 = df['pad_idx2'].astype(int).tolist()
        
        self.smiles_list = df['SMILES'].tolist()       
        self.uniprot_id1 = df['uniprot_id1'].tolist()
        self.uniprot_id2 = df['uniprot_id2'].tolist()
        
        self.label_list = torch.FloatTensor(df['label'].tolist())

        self.process_compound()
        self.process_protein()
        
        self.df = df

        
    def process_compound(self):
        with open('data/features/compound_phy.pickle', 'rb') as f:
            compound_phy = pickle.load(f)
        with open('data/features/src_tokens_dict.pickle', 'rb') as f:
            src_tokens = pickle.load(f)
        with open('data/features/src_distance_dict.pickle', 'rb') as f:
            src_distance = pickle.load(f)
        with open('data/features/src_coord_dict.pickle', 'rb') as f:
            src_coord = pickle.load(f)
        with open('data/features/src_edge_type_dict.pickle', 'rb') as f:
            src_edge_type = pickle.load(f)

        self.compound_phy = {k: torch.Tensor(v).float().to(self.device) for k, v in compound_phy.items()}
        self.src_tokens = {k: torch.Tensor(v).long().to(self.device) for k, v in src_tokens.items()}
        self.src_distance = {k: torch.Tensor(v).float().to(self.device) for k, v in src_distance.items()}
        self.src_coord = {k: torch.Tensor(v).float().to(self.device) for k, v in src_coord.items()}
        self.src_edge_type = {k: torch.Tensor(v).long().to(self.device) for k, v in src_edge_type.items()}
      

    def process_protein(self):
        with open('data/features/esm2.pickle', 'rb') as f:
            esm = pickle.load(f)
        protein_phy = pd.read_csv('data/features/protein_phy.csv')
        kg = pd.read_csv('data/features/knowledge_graph_embedding.csv')

        self.esm = {k: torch.FloatTensor(v).to(self.device) for k, v in esm.items()}
        self.kg = {kg.uniprot_id[i]: torch.FloatTensor(kg.iloc[i, 1:]).to(self.device) for i in range(len(kg))}
        self.protein_phy = {protein_phy.uniprot_id[i]: torch.FloatTensor(protein_phy.iloc[i, 1:]).to(self.device) for i in range(len(protein_phy))}

        
    def truncate_sequence(self, esm, beg_idx, end_idx, pad_idx):
        esm = esm[beg_idx:end_idx+1, :]
        if pad_idx != 0:
            esm = F.pad(esm, (0, 0, 0, pad_idx))
        return esm
    
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        uniprot_id1 = self.uniprot_id1[idx]
        uniprot_id2 = self.uniprot_id2[idx]
        
        src_tokens = self.src_tokens[smiles]
        src_distance = self.src_distance[smiles]
        src_coord = self.src_coord[smiles]
        src_edge_type = self.src_edge_type[smiles]
        esm1 = self.truncate_sequence(self.esm[uniprot_id1], self.beg_idx1[idx], self.end_idx1[idx], self.pad_idx1[idx])
        esm2 = self.truncate_sequence(self.esm[uniprot_id2], self.beg_idx2[idx], self.end_idx2[idx], self.pad_idx2[idx])
        auxiliary = torch.cat((self.protein_phy[uniprot_id1], self.kg[uniprot_id1], 
                               self.protein_phy[uniprot_id2], self.kg[uniprot_id2], 
                               self.compound_phy[smiles]), 0)
        
        num_resi = torch.IntTensor([self.beg_idx1[idx], self.end_idx1[idx], self.pad_idx1[idx], 
                                    self.beg_idx2[idx], self.end_idx2[idx], self.pad_idx2[idx]])
        label = self.label_list[idx]
        
        return src_tokens, src_distance, src_coord, src_edge_type, esm1, esm2, auxiliary, num_resi, label


    def __len__(self):
        return len(self.label_list)
    
    
    def get_df(self):
        return self.df
    
    

    
class PPIInhibitorInferenceDataset():
    def __init__(self, datapath, device, maxlen=800):
        self.device = device
        
        df = pd.read_csv(datapath).drop(columns=['uniprot_id1', 'uniprot_id2'])
        interface = pd.read_csv('data/processed_interface.csv')
        df = pd.merge(df, interface, left_on='ppi_label', right_on='ppi_label')
        
        self.beg_idx1 = df['beg_idx1'].astype(int).tolist()
        self.end_idx1 = df['end_idx1'].astype(int).tolist()
        self.pad_idx1 = df['pad_idx1'].astype(int).tolist()
        self.beg_idx2 = df['beg_idx2'].astype(int).tolist()
        self.end_idx2 = df['end_idx2'].astype(int).tolist()
        self.pad_idx2 = df['pad_idx2'].astype(int).tolist()
        
        self.smiles_list = df['SMILES'].tolist()       
        self.uniprot_id1 = df['uniprot_id1'].tolist()
        self.uniprot_id2 = df['uniprot_id2'].tolist()
        
        self.label_list = torch.FloatTensor([[1]]*len(df)) #temporary

        self.process_compound()
        self.process_protein()
        
        self.df = df

        
    def process_compound(self):
        with open('data/toy_example/compound_phy.pickle', 'rb') as f:
            compound_phy = pickle.load(f)
        with open('data/toy_example/src_tokens_dict.pickle', 'rb') as f:
            src_tokens = pickle.load(f)
        with open('data/toy_example/src_distance_dict.pickle', 'rb') as f:
            src_distance = pickle.load(f)
        with open('data/toy_example/src_coord_dict.pickle', 'rb') as f:
            src_coord = pickle.load(f)
        with open('data/toy_example/src_edge_type_dict.pickle', 'rb') as f:
            src_edge_type = pickle.load(f)

        self.compound_phy = {k: torch.Tensor(v).float().to(self.device) for k, v in compound_phy.items()}
        self.src_tokens = {k: torch.Tensor(v).long().to(self.device) for k, v in src_tokens.items()}
        self.src_distance = {k: torch.Tensor(v).float().to(self.device) for k, v in src_distance.items()}
        self.src_coord = {k: torch.Tensor(v).float().to(self.device) for k, v in src_coord.items()}
        self.src_edge_type = {k: torch.Tensor(v).long().to(self.device) for k, v in src_edge_type.items()}
      

    def process_protein(self):
        with open('data/features/esm2.pickle', 'rb') as f:
            esm = pickle.load(f)
        protein_phy = pd.read_csv('data/features/protein_phy.csv')
        kg = pd.read_csv('data/features/knowledge_graph_embedding.csv')

        self.esm = {k: torch.FloatTensor(v).to(self.device) for k, v in esm.items()}
        self.kg = {kg.uniprot_id[i]: torch.FloatTensor(kg.iloc[i, 1:]).to(self.device) for i in range(len(kg))}
        self.protein_phy = {protein_phy.uniprot_id[i]: torch.FloatTensor(protein_phy.iloc[i, 1:]).to(self.device) for i in range(len(protein_phy))}

   
    def truncate_sequence(self, esm, beg_idx, end_idx, pad_idx):
        esm = esm[beg_idx:end_idx+1, :]
        if pad_idx != 0:
            esm = F.pad(esm, (0, 0, 0, pad_idx))
        return esm
    
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        uniprot_id1 = self.uniprot_id1[idx]
        uniprot_id2 = self.uniprot_id2[idx]
        
        src_tokens = self.src_tokens[smiles]
        src_distance = self.src_distance[smiles]
        src_coord = self.src_coord[smiles]
        src_edge_type = self.src_edge_type[smiles]
        esm1 = self.truncate_sequence(self.esm[uniprot_id1], self.beg_idx1[idx], self.end_idx1[idx], self.pad_idx1[idx])
        esm2 = self.truncate_sequence(self.esm[uniprot_id2], self.beg_idx2[idx], self.end_idx2[idx], self.pad_idx2[idx])
        auxiliary = torch.cat((self.protein_phy[uniprot_id1], self.kg[uniprot_id1], 
                               self.protein_phy[uniprot_id2], self.kg[uniprot_id2], 
                               self.compound_phy[smiles]), 0)
        
        num_resi = torch.IntTensor([self.beg_idx1[idx], self.end_idx1[idx], self.pad_idx1[idx], 
                                    self.beg_idx2[idx], self.end_idx2[idx], self.pad_idx2[idx]])
        label = self.label_list[idx]
        
        return src_tokens, src_distance, src_coord, src_edge_type, esm1, esm2, auxiliary, num_resi, label


    def __len__(self):
        return len(self.label_list)
    
    
    def get_df(self):
        return self.df