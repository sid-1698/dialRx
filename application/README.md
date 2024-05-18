## Setup Instructions

### Models

1. Download `BioMistral-7B.Q4_K_M.gguf` from [here](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF) and place it in the `models` folder.
2. Download `meditron-7b.Q4_K_M.gguf` from [here](https://huggingface.co/TheBloke/meditron-7B-GGUF) and place it in the `models` folder.

### Qdrant (VectorDB)

1. Setup Qdrant in your local system using the [quick start guide](https://qdrant.tech/documentation/quick-start/).
2. Download the `qdrant_storage` folder from [this link](https://gtvault-my.sharepoint.com/:f:/g/personal/vsanku6_gatech_edu/EiFdSSsNp6VEp7oQxWVJiHQB9EZzxf_jup85Jdm-ljL_GQ?e=TF4UH1) and place it under the root folder. It contains the database collections with vector data from CSV files.

### Python Environment

1. Create a Python environment using the `requirements.txt` file.

## Starting Application

1. **UI**: Go to the `ui` folder and run `streamlit run app.py`.
2. **Backend**: run `python app.py` from root, it starts a Flask server.
3. **Qdrant**: Run the following command (make sure it points to the `qdrant_storage` folder downloaded earlier):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
