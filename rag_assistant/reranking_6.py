from google import genai
from sentence_transformers import CrossEncoder
import json
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from hybrid_retrival_5 import get_top_10


load_dotenv()
client = genai.Client()

def rerank(query: str):
    rankings = get_top_10(query)
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    dbclient = chromadb.PersistentClient(path="./manual_db")
    collection = dbclient.get_collection(name="manual_vectors")
    combined_list = []
    result = collection.get(list(rankings.keys()))["documents"]
    for r in result:
        combined_list.append((query, r))
    prediction = model.predict(combined_list)
    new_rankings = {}
    i = 0
    for id in rankings.keys():
        new_rankings[id] = prediction[i].item()
        i += 1
    top_3 = dict(sorted(new_rankings.items(), key=lambda item:[item[1]], reverse=True)[:3])
    return top_3

if __name__ == "__main__":
    user_question = input("What question would you like to ask the database?\n> ")
    rerank(user_question)