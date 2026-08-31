from google import genai
import json
from tenacity import retry, wait_fixed, stop_after_attempt
import uuid
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
client = genai.Client()

@retry(
    wait=wait_fixed(65),
    stop=stop_after_attempt(5)
)
def get_embedding(text: str) -> list[float]:
        response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    # The API returns a deeply nested object; we just want the list of floats
        return response.embeddings[0].values

def main():
    file = Path("chunks.json")
    database = []
    if file.is_file() and file.stat().st_size != 0:
        with open("chunks.json", "r") as f:
            chunks = json.load(f)
            
        for i in range(len(chunks)):
            chunk = chunks[i]
            database.append({
            "id": str(uuid.uuid4()),
            "text": chunk,
            "vector": get_embedding(chunk),
            "metadata": {"source": "joke-structure-guide.pdf", 
                        "chunk_index": i,
                        "char_count": len(chunk)}})
        
        with open("vector_payload.json", "w") as f:
                json.dump(database, f, indent=4)
    
    else:   
        print("File not created. Please run text_chunker to create the file.")
    
if __name__ == "__main__":
    main()
