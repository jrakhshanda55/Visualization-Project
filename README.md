# Software Architecture Analysis Dashboard

An **interactive dashboard** for exploring and visualizing software architecture using Graph Neural Networks (GNNs) and Graph Autoencoders (GAEs). The app allows you to analyze dependencies, view architectural structures, and train GNN-based models to learn software module representations.

### Features

- Load and explore software dependency datasets  
- Visualize implemented and intended architectures  
- View module and dependency type distributions  
- Train **GAE** or **VGAE** models using **GCN** or **GAT** encoders  
- Compare 2D embedding projections using **UMAP** or **t-SNE**  

---

## Requirements and Running the Dashboard

All dependencies are listed in `requirements.txt`.  
Install them with:

```bash
pip install -r requirements.txt
```
Then run the Dashboard on console with:
```bash
python app.py
```
## Project Structure

```bash
visualization_project/
│
├── app.py                       
├── requirements.txt        
├── compute/
│   ├── data_builder.py       
│   ├── gnn_models.py            
│   ├── projection.py             
│   └── plot_architecture.py      
└── data/
    ├── <dataset_name>.csv       
    ├── <dataset_name>_deps.csv  
    └── <dataset_name>_relations.csv (optional intended architecture)

```
