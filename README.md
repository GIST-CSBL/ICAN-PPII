# ICAN-PPII: Interpretable cross-attention network for predicting protein-protein interaction inhibitors 

## Abstract
Protein-protein interactions (PPIs) are crucial therapeutic targets, yet discovering small molecule inhibitors remains challenging due to the complexity of protein-protein interfaces. We present an interpretable AI model based on a cross-attention mechanism that effectively predicts PPI inhibitors by leveraging sequence-based information of protein complexes. Our model integrates multiple data representations: pre-trained embeddings of protein sequences and molecular structures, knowledge graph embeddings capturing biological context, and key physicochemical properties. A novel cross-attention mechanism enables the model to learn meaningful intermolecular relationships while providing interpretable insights into interaction hotspots between protein interfaces and small molecules. Through comprehensive validation against established benchmarks, we demonstrate that our approach achieves superior predictive performance compared to existing methods. The model's interpretability reveals key molecular recognition patterns, potentially accelerating the rational design of PPI inhibitors. This work represents a significant advancement in computational drug discovery, offering a powerful and practical tool for identifying novel therapeutic candidates targeting protein-protein interactions.


## Usage

### Install dependencies
Set up an Anaconda environment and install required package dependencies.

```
conda create -n ICAN-PPII python=3.9
conda activate ICAN-PPII
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install fair-esm unimol_tools huggingface_hub pandas scikit-learn
```


### Prediction
Currently, predictions are limited, as the knowledge graph has not yet been published. 
Once available, you will be able to predict PPI inhibitors with your own dataset by referring to the `predict.ipynb` file.  

Before making predictions, you need to download pfeature for extracting protein physicochemical properties.
```
git clone https://github.com/GIST-CSBL/ICAN-PPII.git
cd ICAN-PPII/src
git clone https://github.com/raghavagps/Pfeature.git

mkdir weights
cd weights
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt
#reference: https://github.com/facebookresearch/esm
```

Also, you need to download the `weights.model` file from the link below and move it to the `weights/` directory.  
https://drive.google.com/file/d/1Ye5R8-CDkE1a_neF029dnLbY5yQ2Qrpk/view?usp=sharing




## Contact
- Dongok Nam (dongoknam@gm.gist.ac.kr)
- Hojung Nam (Corresponding author) (hjnam@gist.ac.kr)
