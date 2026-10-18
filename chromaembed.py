from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb
chroma_client = chromadb.Client()

# switch \`create_collection\` to \`get_or_create_collection\` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

default_ef = DefaultEmbeddingFunction()
embeddings = default_ef(["foo"])
print(len(embeddings[0])) # [[0.05035809800028801, 0.0626462921500206, -0.061827320605516434...]]

#collection.query(query_embeddings=embeddings)