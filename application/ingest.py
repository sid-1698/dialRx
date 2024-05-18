import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.document_loaders.csv_loader import CSVLoader
from qdrant_client import QdrantClient


embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings")

print(embeddings)

loader = DirectoryLoader('./ingest-data', glob="**/*.csv",
                         show_progress=True, loader_cls=CSVLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=1000)

texts = text_splitter.split_documents(documents)

print(texts[1])


url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="drugs_db_ver_2"
)

print("Vector DB Successfully Created!")
