import chromadb

client = chromadb.PersistentClient(path="chroma_db")

knowledge = client.get_collection("knowledge_collection")
user = client.get_collection("user_collection")

print("Knowledge :", knowledge.count())
print("User      :", user.count())