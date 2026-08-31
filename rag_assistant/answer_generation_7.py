from google import genai
from dotenv import load_dotenv
import chromadb
from reranking_6 import rerank

load_dotenv()
client = genai.Client()


def generate(question: str):
    top_3 = rerank(question)
    system_prompt = "Using only the chunks provided, answer the question the user asks. \
    If you are unable to answer using the chunks or , tell the user that you are unable to answer\
    the question. Each chunk has an id associated with it, coming just before the block of text. Each chunk and id is separated by ---\
    When you provide an answer, make sure to append the id of the chunk you used \
    at the end of the relevant sentence. Question: "
    chunks = ""
    dbclient = chromadb.PersistentClient(path="./manual_db")
    collection = dbclient.get_collection(name="manual_vectors")
    ids = list(top_3.keys())
    result = collection.get(ids)["documents"]
    for i in range(len(ids)):
        chunks += " Chunk ID: " + ids[i] + " Text: " + result[i]  + " --- "
    
    system_prompt += question + chunks
    response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=system_prompt)
    return response.text, top_3


if __name__ == "__main__":
    user_question = input("What question would you like to ask the database?\n> ")
    generate(user_question)