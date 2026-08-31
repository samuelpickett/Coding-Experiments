from google import genai
from dotenv import load_dotenv
from reranking_6 import rerank
import chromadb

load_dotenv()
client = genai.Client()


chat_history = []
def reduce_chat_log(chat: list[(str, str)]):
    if len(chat_history) > 5:
        chat.pop(0)


def main():
    dbclient = chromadb.PersistentClient(path="./manual_db")
    collection = dbclient.get_collection(name="manual_vectors")
    
    while True:
        # Takes the user input and makes it easier for the RAG to understand
        user_input = input("> ")
        sys_prompt = """Using the provided chat log, rewrite the user's question to a standalone, search-optimized question.
        The chat log is a list of (str, str) tuples, where the first string is the user input and the second is the resulting answer
        Only give back the rewritten input and nothing else. User Question:""" + user_input + "Chat History: " + str(chat_history)
        response = client.models.generate_content(model="gemini-3.5-flash", contents=sys_prompt)
        
        # Gets the chunks from the database and matches them with their id. 
        ranks = rerank(response.text)
        chunks = ""
        ids = list(ranks.keys())
        result = collection.get(ids)["documents"]
        for i in range(len(ids)):
            chunks += " Chunk ID: " + ids[i] + " Text: " + result[i]  + " --- "
            
        final_prompt = """Using only the chunks provided, answer the question the user asks.
        If you are unable to answer using the chunks, tell the user that you are unable to answer
        the question without mentioning the chunks. Each chunk has an id associated with it, coming just before the block of text. Each chunk and id is separated by ---
        When you provide an answer, make sure to append the id of the chunk you used 
        at the end of the relevant sentence. You are also provided the chat history in the form of a list of (str, str) tuples where the first
        string is the user input and the second is the response. Use the chat history to help make the conversation feel natural. If they don't ask a question, 
        continue the conversation in a normal and natural way while still only referring to the chunks for information. Chat History: 
        """ + str(chat_history) + ". Chunks:" + chunks + "User question: " + user_input
        response = client.models.generate_content(model="gemini-3.5-flash",
                        contents=final_prompt)
        print(response.text)
        chat_history.append((user_input, response.text))
        reduce_chat_log(chat_history)
    
    
if __name__ == "__main__":
    main()