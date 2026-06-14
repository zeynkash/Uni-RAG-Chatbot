import faiss
import numpy as np
import os

print("Loading embeddings...")
embeddings = np.load("chunking/embeddings_all_universities.npy").astype('float32')

print("Normalizing embeddings for cosine similarity...")
faiss.normalize_L2(embeddings)

print("Building FAISS index...")
index = faiss.IndexFlatIP(1536)
index.add(embeddings)

print("Saving FAISS index...")
faiss.write_index(index, "chunking/faiss_index_all_universities.bin")
print("Done!")
