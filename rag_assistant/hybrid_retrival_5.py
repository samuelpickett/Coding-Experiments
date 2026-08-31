from google import genai
import chromadb
import rank_bm25
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

def get_top_10(query: str):
    dbclient = chromadb.PersistentClient(path="./manual_db")
    collection = dbclient.get_collection(name="manual_vectors")
    all_data = collection.get()
    text = all_data["documents"]
    ids = all_data["ids"]
    tokenized_text = [doc.lower().split(" ") for doc in text]
    ranker = rank_bm25.BM25Okapi(tokenized_text)
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query)
    result = collection.query(response.embeddings[0].values)
    tokenized_question = query.lower().split(" ")
    doc_scores = ranker.get_scores(tokenized_question)
    top_10_indices = sorted(
    range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
    )[:10]
    
    rrf_scores = {}
    i = 1
    for id in result["ids"][0]:
        rrf_scores[id] = 1 / (60 + i)
        i += 1

    i = 1
    for id in top_10_indices:
        if ids[id] not in rrf_scores:
            rrf_scores[ids[id]] = 1 / (60 + i)
        else:
            rrf_scores[ids[id]] += 1 / (60 + i)
        
        i += 1
    top_10 = dict(sorted(rrf_scores.items(), key=lambda item:[item[1]], reverse=True)[:10])
    return top_10
    
    
if __name__ == "__main__":
    user_question = input("What question would you like to ask the database?\n> ")
    get_top_10(user_question)