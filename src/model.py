import torch
from torch import nn
import torch.nn.functional as F


class PPIInhibitorModel(nn.Module):
    def __init__(self, compound_model, dropout=0.2):
        super(PPIInhibitorModel, self).__init__()
        
        self.inf = -5.0E10
        self.compound_model = compound_model
        
        self.protein_fc = nn.Linear(640, 64)
        self.compound_fc = nn.Linear(512, 64)
        
        self.contextual_fc = nn.Linear(384, 128)
        self.auxiliary_fc = nn.Linear(86, 64)
        
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        
    def pad_compound(self, compound):
        size = max(c.size(0) for c in compound)
        compound_padded = []
        for i in range(len(compound)):
            compound_padded.append(F.pad(compound[i], (0, 0, 0, size-len(compound[i]))))
        compound_padded = torch.stack(compound_padded)
        return compound_padded
        
        
    def pad_similarity_score(self, similarity_score, num_atoms, num_resi, ppi_attention):
        num = self.inf
        if ppi_attention:
            for i in range(similarity_score.shape[0]):
                similarity_score[i, :num_resi[i][0]] = num
                similarity_score[i, num_resi[i][1]:] = num
                similarity_score[i, :, :num_resi[i][2]] = num
                similarity_score[i, :, num_resi[i][3]:] = num
        else:
            for i in range(similarity_score.shape[0]):
                similarity_score[i, num_atoms[i]:] = num
                similarity_score[i, :, :num_resi[i][0]] = num
                similarity_score[i, :, num_resi[i][1]:] = num
        return similarity_score
    
    
    def calculate_attention(self, xd, xt, batch_num_objs, num_resi, ppi_attention=False):
        similarity_score = torch.bmm(xd, xt)
        similarity_score = self.pad_similarity_score(similarity_score, batch_num_objs, num_resi, ppi_attention)
        
        s_a = F.softmax(similarity_score, dim=1)
        a_s = F.softmax(similarity_score, dim=-1)
        
        s_a = s_a.permute(0, 2, 1)
        s_a_xd = torch.bmm(s_a, xd)
        feature1 = torch.sum(s_a_xd, 1)
        
        xt = xt.permute(0, 2, 1)
        a_s_xt = torch.bmm(a_s, xt)
        feature2 = torch.sum(a_s_xt, 1)
        
        feature = torch.cat([feature1, feature2], 1)
        return feature
    

    def forward(self, src_tokens, src_distance, src_coord, src_edge_type, esm1, esm2, auxiliary, num_resi):
        # inter-contexutal representation
        compound = self.compound_model(src_tokens, src_distance, src_coord, src_edge_type, return_repr=True, return_atomic_reprs=True)
        compound = compound['atomic_reprs']
        num_atoms = [c.size(0) for c in compound]
        compound = self.pad_compound(compound)
        
        ## cross attention module
        compound = self.compound_fc(compound)
        esm1 = self.protein_fc(esm1)
        esm2 = self.protein_fc(esm2)
        
        dti1 = self.calculate_attention(compound, esm1.permute(0, 2, 1), num_atoms, num_resi[:, :2])  
        dti2 = self.calculate_attention(compound, esm2.permute(0, 2, 1), num_atoms, num_resi[:, 2:]) 
        ppi = self.calculate_attention(esm1, esm2.permute(0, 2, 1), num_atoms, num_resi, ppi_attention=True) 
        
        contextual = torch.cat((dti1, dti2, ppi), 1)
        contextual = self.contextual_fc(contextual)
        

        # auxiliary information
        auxiliary = self.auxiliary_fc(auxiliary)

        
        # prediction module
        x = torch.cat((contextual, auxiliary), 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.out(x)

        return x