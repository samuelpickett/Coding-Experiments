from dotenv import load_dotenv
from google import genai
import numpy as np
from pathlib import Path
import nltk
import json
from tenacity import retry, wait_fixed, stop_after_attempt

load_dotenv()
client = genai.Client()



@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(65)
)
def get_embedding(text: str) -> list[float]:
        response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    # The API returns a deeply nested object; we just want the list of floats
        return response.embeddings[0].values

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    with open("parsed_document.md", "r", encoding="utf-8") as file:
        text = file.read()
    sentences = nltk.sent_tokenize(text)
    md_embeddings = []
    chunks = []
    # Get embeddings for each sentence
    file = Path("embeddings.json")
    if file.is_file() and file.stat().st_size != 0:
            with open("embeddings.json", "r") as f:
                md_embeddings = json.load(f)
    else:
        for i in range(len(sentences)):
            md_embeddings.append((sentences[i], get_embedding(sentences[i])))
            print(i)
    
    with open("embeddings.json", "w") as f:
        json.dump(md_embeddings, f, indent=4)
    
    # Sets the first chunk as the first sentence in the embeddings list
    chunk = md_embeddings[0][0]
    for i in range(len(md_embeddings)):
        if i < len(md_embeddings) - 1:
            distance = 1 - cosine_similarity(md_embeddings[i][1], md_embeddings[i + 1][1])
            # If the distance between two embeddings is greater than .3, add the current chunk to 
            # the list and set chunk to the next sentence
            if distance > .4:
                chunks.append(chunk)
                chunk = md_embeddings[i + 1][0]
            # Otherwise, keep adding sentences to the chunk
            else:
                chunk += " " + md_embeddings[i + 1][0]
        else:
            distance = 1 - cosine_similarity(md_embeddings[i][1], md_embeddings[i - 1][1])
            # If the distance between two embeddings is greater than .3, add the current chunk to 
            # the list and set chunk to the next sentence
            if distance > .4:
                chunks.append(chunk)
                chunk = md_embeddings[i][0]
            # Otherwise, keep adding sentences to the chunk
            else:
                chunk += " " + md_embeddings[i][0]
            chunks.append(chunk)

    # If chunks are less than 200 characters, add the next chunk to it and remove the next chunk
    merged_chunks = []
    current_string = ""
    for chunk in chunks[:-1]:
        if len(current_string) < 200:
            current_string += " " + chunk
        else:
            merged_chunks.append(current_string)
            current_string = chunk
    if len(chunks[-1]) < 200 and len(chunks) > 0:
        merged_chunks[-1] += " " + chunks[-1]
    else:
        merged_chunks.append(chunks[-1])
    
    print(len(merged_chunks))
    with open("chunks.json", "w") as f:
            json.dump(merged_chunks, f, indent=4)




if __name__ == "__main__":
    main()
