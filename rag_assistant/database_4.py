import json
import chromadb
from pathlib import Path



def main():
    file = Path("vector_payload.json")
    if file.is_file() and file.stat().st_size != 0:
        with open("vector_payload.json", "r") as f:
            database = json.load(f)    
        client = chromadb.PersistentClient(path="./manual_db")
        collection = client.get_or_create_collection(name="manual_vectors")
        database_lists = {key: [d[key] for d in database] for key in database[0]}
        ids = database_lists["id"]
        text = database_lists["text"]
        vectors = database_lists["vector"]
        metadata = database_lists["metadata"]
        collection.add(ids=ids, embeddings=vectors, metadatas=metadata, documents=text)
    else:
        print("There is no vector payload to upload. Please perform the previous operations before using this program again.")
        


if __name__ == "__main__":
    main()
