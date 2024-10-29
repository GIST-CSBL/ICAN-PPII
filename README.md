# ICAN-PPII: Interpretable cross-attention network for predicting protein-protein interaction inhibitors 

## Abstract
Protein-protein interactions (PPIs) are crucial therapeutic targets, yet discovering small molecule inhibitors remains challenging due to the complexity of protein-protein interfaces. We present an interpretable AI model based on a cross-attention mechanism that effectively predicts PPI inhibitors by leveraging sequence-based information of protein complexes. Our model integrates multiple data representations: pre-trained embeddings of protein sequences and molecular structures, knowledge graph embeddings capturing biological context, and key physicochemical properties. A novel cross-attention mechanism enables the model to learn meaningful intermolecular relationships while providing interpretable insights into interaction hotspots between protein interfaces and small molecules. Through comprehensive validation against established benchmarks, we demonstrate that our approach achieves superior predictive performance compared to existing methods. The model's interpretability reveals key molecular recognition patterns, potentially accelerating the rational design of PPI inhibitors. This work represents a significant advancement in computational drug discovery, offering a powerful and practical tool for identifying novel therapeutic candidates targeting protein-protein interactions.


## Usage

### Install dependencies
Set up an Anaconda environment and install required package dependencies.

```
conda create -n PPIINet python=3.9
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install unimol_tools
pip install pandas, numpy, scikit-learn
```

### Prediction
To predict PPI inhibitors with your own dataset, please refer to the `predict.ipynb` file.  
Before making predictions, you need to download the `weights.model` file from the link below and move it to the `weights/` directory.  
https://drive.google.com/file/d/1Ye5R8-CDkE1a_neF029dnLbY5yQ2Qrpk/view?usp=sharing

Currently, predictions are limited to PPI targets included in the training dataset, as the knowledge graph has not yet been published.

### Train
To reproduce the results or fine-tune the model, please refer to the train.ipynb file.  
To reproduce the results, download the preprocessed feature files (.pickle) from the link below and place them in the data/features/ directory.  
https://drive.google.com/drive/folders/1IJU6l0wvAQOKQvm3EJ492zG2j39sOKlS?usp=sharing



## Contact
- Dongok Nam (dongoknam@gm.gist.ac.kr)
- Hojung Nam (Corresponding author) (hjnam@gist.ac.kr)
