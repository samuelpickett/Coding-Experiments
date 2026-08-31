import math
import json
from dotenv import load_dotenv
from pathlib import Path
from google import genai
from tenacity import Retrying, stop_after_attempt, wait_exponential


load_dotenv()
client = genai.Client()

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

def get_embedding(text: str) -> list[float]:
        for attempt in Retrying(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2)):
            with attempt:
                response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text
    )
    # The API returns a deeply nested object; we just want the list of floats
        return response.embeddings[0].values

def chunk_text(text: str, chunk_size:int=500, overlap:int=100) ->list[str]:
    """
    Splits a long string into smaller text chunks with a sliding window. 

    Args:
        text (str): The text you want to chunk
        chunck_size (int, optional): How many characters long you want the chunks to be. Defaults to 500.
        overlap (int, optional): How many characters will overlap between the chunks. Defaults to 100.

    Returns:
        list[str]: A list of strings of length chunk_size
    """
    chunks = []
    start = 0
    # Checks the size of the overlap to prevent an infinite loop
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size.")
    
    while start < len(text):
        # Calculates end index for current chunk
        end = start + chunk_size
        # Grabs the chunk and adds it to the end of the list
        chunk = text[start:end]
        chunks.append(chunk)
        # Moves the starting window forward
        start += chunk_size - overlap
    
    return chunks



def main():
    documents = [
        "Professor Smith's Pyhsics 201 midterm is scheduled for October 14th.",
        "The physics lab requires closed-toe shoes and safety goggles at all times.",
        "Office hours for Physics 201 are Tuesdays and Thursdays from 2 PM to 4 PM.",
        "Late homework assignments lose 10% of their grade per day late."
    ]
    
    file = Path("syllabus_embeddings.json")
    syllabus_embeddings = []
    print("Embedding syllabus into the vector database...")
    
    with open("syllabus.txt", encoding="utf-8") as f:
        syllabus = f.read()

    syllabus_chunks = chunk_text(syllabus)
    
    if file.is_file() and file.stat().st_size != 0:
        with open("syllabus_embeddings.json", "r") as f:
            syllabus_embeddings = json.load(f)
            
    else:
        for chunks in syllabus_chunks:
            vector = get_embedding(chunks)
            syllabus_embeddings.append({"text": chunks, "vector": vector})
            
        with open("syllabus_embeddings.json", "w") as f:
            json.dump(syllabus_embeddings, f, indent=4)
    
    user_query = input("What question would you like answered from the syllabus?\n> ")
    print(f"\nUser asks: {user_query}")
    
    query_vector = get_embedding(user_query)
    best_match = ""
    highest_score = -1.0
    
    for item in syllabus_embeddings:
        score = cosine_similarity(query_vector, item["vector"])
        if score > highest_score:
            highest_score = score
            best_match = item["text"]
    print(f"\nRetrieved Context: {best_match} (Score: {highest_score:.4f})")
    
    
    rag_prompt = f"""You are a hepful college teaching assistant. Answer the user's question using ONLY the provided context below. 
    If the context does not contain the answer, say I don't know based on the syllabus
    
    Context:
    {best_match}
    
    User Question:
    {user_query}
    """
    
    
    print("\nSending augmented prompt to Gemini...\n")
    for attempt in Retrying(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2)):
            with attempt:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=rag_prompt
    )
    print("--- Final AI Response ---")
    print(response.text)
    print("-------------------------")

if __name__ == "__main__":
    main()