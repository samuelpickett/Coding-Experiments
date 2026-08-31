from dotenv import load_dotenv
from google import genai
import time
import datetime

load_dotenv()



def main():
    client = genai.Client()
    client2 = genai.Client()
    endtime = datetime.datetime.now() + datetime.timedelta(minutes=10)
    max_retries = 5
    retries = 0
    ai1prompt = "I want to do a little roleplay. You will play the part of a\
        first time user of AI. Come up with a random thing you want to ask the AI\
        about. Each time the AI responds, come up with a new prompt to ask based off of\
        the response."
    # print("What do you want to ask Gemini? (Enter 0 to end the conversation)\n> ", end="")
    print("Starting conversation between 2 AIs.")
    while datetime.datetime.now() < endtime and retries < max_retries:
        try: 
            # user_input = input()
            # if user_input == "0":
            #     break
            try:
                ai2prompt = client.models.generate_content(model="gemini-3.5-flash", 
                contents=ai1prompt)
                print("--- AI 1 started with:\n")
                print(ai2prompt.text)
                print("----------")
                retries = 0
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    print("Model is busy. Retrying")
                    time.sleep(2 * (2** retries))
                    retries += 1
                    
            
            try:
                ai1prompt = client2.models.generate_content(model="gemini-3.5-flash", 
                contents=ai2prompt.text)
                print("\n--- AI 2's Response:")
                print(ai2prompt.text)
                print("----------")
                retries = 0
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    print("Model is busy. Retrying")
                    time.sleep(2 * (2** retries))
                    retries += 1
        
        except Exception as e:
            if "11001" in str(e):
                print("Please check your internet connection.")
                break



    if retries == max_retries:
        print("Too many people are using the model. Please restart the program.")
if __name__ == "__main__":
    main()