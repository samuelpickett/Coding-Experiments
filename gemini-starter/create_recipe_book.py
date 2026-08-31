from dotenv import load_dotenv
from recipes import Recipe
from google import genai
import time
load_dotenv()

client = genai.Client()

def ask_gemini(model_name, prompt, max_retries=5, config=None):
    base_wait_time = 2
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            return response.text

        except Exception as e:
            if "503" in str(e) and attempt < max_retries:
                wait = base_wait_time ** attempt
                print(f"Model is busy. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e



def main():
    for _ in range(1):
        
        ai_recipe = ask_gemini("gemini-3.5-flash", "Give me a recipe for a breakfast meal.")
        print(ai_recipe)
        
        ai_extrated = ask_gemini("gemini-3.5-flash", "Format the following recipe into schema that will be provided: " + ai_recipe, config={"response_schema":Recipe})
        print(ai_extrated)
    
    
    
    
    



if __name__ == "__main__":
    main()