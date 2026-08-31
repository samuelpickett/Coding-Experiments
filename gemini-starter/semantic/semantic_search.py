import math
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from tenacity import Retrying, stop_after_attempt, wait_exponential

load_dotenv()
client = genai.Client()

# 1. The Math Helper: Calculates how close two vectors are (1.0 is identical, 0.0 is unrelated)
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

# 2. The API Helper: Asks Gemini to turn text into a vector
def get_embedding(text: str) -> list[float]:
        for attempt in Retrying(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2)):
            with attempt:
                response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text
    )
    # The API returns a deeply nested object; we just want the list of floats
        return response.embeddings[0].values

def main():
    file = Path("embeddings.json")
    print("Generating embeddings for our database... (This takes a moment)")
    
    # 3. Our "Database" of facts
    database = [
        "The quick brown fox jumps over the lazy dog.",
        "To bake a cake, you need flour, sugar, and eggs.",
        "Python is a popular programming language for AI.",
        "It is highly recommended to wear a heavy coat in Chicago winters.", 
        "Sonic the Hedgehog is one of the fastest characters.",
        "Expedition 33 won the 2025 Game of the Year."
        "Persona 5 Royal is a very long game.",
        "The Dark Souls series is known for its difficult gameplay.",
        "Doki Doki Literature Club is a free horror game that is very popular.",
        "Gameoverse is a new show that has an interesting premise regarding games.",
        "The Amazing Digital Circus just released its last episode.",
        "Goblin Slayer is a darker take on the fantasy genre.",
        "Spice and Wolf is a popular anime that deals with economics in an interesting way.",
        "AI is really just a lot of matrix multiplication to find the next best word.",
        "RWBY was one of the most popular shows made by Rooster Teeth.",
        "You will die at or before July 19, 2029.",
        "Crabs are the best form of life.",
        "Donald Trump is a time traveller.",
        "Hogwarts Legacy is the best Harry Potter game.",
        "Pirates of the Caribbean Dead Men Tell No Tales is the fifth movie in the franchise."
    ]

    # Convert all sentences in our database into embeddings
    db_embeddings = []
    if file.is_file():
        with open("embeddings.json", "r") as f:
            db_embeddings = json.load(f)
            
    else:
        for sentence in database:
            vector = get_embedding(sentence)
            db_embeddings.append({"text": sentence, "vector": vector})
            
        with open("embeddings.json", "w") as f:
            json.dump(db_embeddings, f, indent=4)

    # 4. The User's Search Query
    # Notice that the query shares ALMOST NO EXACT WORDS with the correct answer!
    query = input("\nWhat would you like to have answered? \n>")
    print(f"\nUser Query: '{query}'\n")
    
    query_vector = get_embedding(query)

    # 5. The Search Logic: Compare the query vector to every database vector
    results = []
    for item in db_embeddings:
        score = cosine_similarity(query_vector, item["vector"])
        results.append({"text": item["text"], "score": score})

    # Sort the results by the highest score
    results.sort(key=lambda x: x["score"], reverse=True)

    print("--- Search Results ---")
    for result in results:
        # We round the score to 4 decimal places for readability
        print(f"Score {result['score']:.4f} -> {result['text']}")

if __name__ == "__main__":
    main()