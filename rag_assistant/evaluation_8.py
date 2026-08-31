from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate, RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from answer_generation_7 import generate
from google.genai.errors import APIError
import chromadb

load_dotenv()


def eval():
    judge = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    dbclient = chromadb.PersistentClient(path="./manual_db")
    collection = dbclient.get_collection(name="manual_vectors")
    #  "Where am I?", "What is a pun?", "How many chapters are in the book?" What is the capital of France?", "Where am I?", 
    tests = {"question": ["What is a pun?"], 
            "contexts": [], 
            "answer": [], 
# "I cannot answer based on the text provided.", "I cannot answer based on the text provided.", "There are 11 chapters in the provided text. "
            "ground_truth": ["A pun, or double-entendre, is when you take a word with multiple meanings and use it in a context where more than one meaning makes sense. It is often considered the lowest form of humor, but can be used to great effect."
            ]}
    
    for question in tests["question"]:
        answer, top_3 = generate(question)
        tests["answer"].append(answer)
        ids = list(top_3.keys())
        result = collection.get(ids)["documents"]
        tests["contexts"].append(result)
        
    dataset = Dataset.from_dict(tests)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ]    
    print(evaluate(dataset, metrics=metrics, llm=judge, embeddings=embedder, run_config=RunConfig(timeout=120, max_retries=15, max_workers=1, max_wait=61, exception_types=APIError)))
    print(tests)
    
if __name__ == "__main__":
    eval()