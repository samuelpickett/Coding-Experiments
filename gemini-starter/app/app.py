import os
import sys

# Get the absolute path to the parent directory (project/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from rag import cosine_similarity, get_embedding, chunk_text
from pathlib import Path
import json
from tenacity import Retrying, stop_after_attempt, wait_exponential




app = FastAPI()
load_dotenv()
client = genai.Client()

# Enable CORS so your local HTML file can talk to your local backend server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the shape of incoming data using Pydantic (From Week 2!)
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_chatbot(request: QueryRequest):
    user_question = request.question
    
    # -------------------------------------------------------------
    # PLACE YOUR WEEK 5 RAG CODE HERE
    # 1. Get embedding of user_question
    # 2. Find best match from your database/file
    # 3. Send augmented prompt to Gemini
    # -------------------------------------------------------------
    file = Path("syllabus_embeddings.json")
    syllabus_embeddings = []
    print("Embedding syllabus into the vector database...")
    
    with open("syllabus.txt", encoding="utf-8") as f:
        syllabus = f.read()

    syllabus_chunks = chunk_text(syllabus, 750)
    
    if file.is_file() and file.stat().st_size != 0:
        with open("syllabus_embeddings.json", "r") as f:
            syllabus_embeddings = json.load(f)
            
    else:
        for chunks in syllabus_chunks:
            vector = get_embedding(chunks)
            syllabus_embeddings.append({"text": chunks, "vector": vector})
            
        with open("syllabus_embeddings.json", "w") as f:
            json.dump(syllabus_embeddings, f, indent=4)

    
    query_vector = get_embedding(user_question)
    best_match = ""
    highest_score = -1.0
    
    for item in syllabus_embeddings:
        score = cosine_similarity(query_vector, item["vector"])
        if score > highest_score:
            highest_score = score
            best_match = item["text"]
    
    # Mocking a response for testing the connection
    rag_prompt = f"""You are a hepful college teaching assistant. Answer the user's question using ONLY the provided context below. 
    If the context does not contain the answer, say I don't know based on the syllabus
    
    Context:
    {best_match}
    
    User Question:
    {user_question}
    """
    
    print(best_match)
    print("\nSending augmented prompt to Gemini...\n")
    for attempt in Retrying(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2)):
            with attempt:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=rag_prompt
    )
    
    return {"response": response.text}
