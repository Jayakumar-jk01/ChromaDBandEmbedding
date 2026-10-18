import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# chroma_client = chromadb.Client()
#to persist the values in db(SQLite) we are using persistent function
chroma_client = chromadb.PersistentClient(path="./db/persistentchromdb")

default_ef = DefaultEmbeddingFunction()

# switch \`create_collection\` to \`get_or_create_collection\` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_stuff",embedding_function=default_ef)

documents=[
    {"id":"doc1", "text":"Hello world"},
    {"id":"doc2", "text":"Paris is my world"},
    {"id":"doc3", "text":"I love maths"},
    {
        "id":"doc4",
        "text":"Google is a product company, it has products like youtube"
    }

]


query="Best video company"

for doc in documents:
    collection.upsert(ids=[doc["id"]],documents=[doc["text"]])


result=collection.query(
    query_texts=[query],
    n_results=1,
    include=["documents", "distances", "embeddings"]

)

# print(result)

for idx,documents in enumerate(result["documents"][0]):
    doc_id=result["ids"][0][idx]
    distance=result["distances"][0][idx]
    embed=result["embeddings"][0][idx]
    print(doc_id)
    print(documents)
    print(distance)
    print(embed)