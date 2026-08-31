from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import Retrying, stop_after_attempt, wait_exponential

load_dotenv()

# Initialize the Gemini client
client = genai.Client()

# STEP 1: Define the actual local Python function
def get_current_weather(location: str) -> dict:
    """Mock function simulating an API call to a live weather database."""
    loc = location.lower()
    if "chicago" in loc:
        return {"temperature": "42°F", "condition": "Windy and freezing"}
    elif "tokyo" in loc:
        return {"temperature": "65°F", "condition": "Rainy"}
    else:
        return {"temperature": "72°F", "condition": "Sunny and clear"}

def get_clothing_recommendation(condition: str) -> str:
    con = condition.lower()
    if "windy and freezing" in con:
        return "Wear a coat"
    elif "rainy" in con:
        return "Bring an umbrella"
    elif "sunny and clear" in con:
        return "Wear a T-Shirt and shorts"

def main():
    # STEP 2: Create a chat session equipped with our tools list
    chat = client.chats.create(
        model="gemini-3.5-flash",
        config=types.GenerateContentConfig(tools=[get_current_weather, get_clothing_recommendation])
    )

    user_prompt = "I'm traveling to Malad right now. Do I need to pack a heavy coat?"
    print(f"User: {user_prompt}\n")

    # STEP 3: Send the user's message to the model
    for attempt in Retrying(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2)):
        with attempt:
            response = chat.send_message(user_prompt)
    print(response)
    # STEP 4: Check if Gemini is asking to use our tool
    if response.function_calls:
        function_call = response.function_calls[0]
        print(f"Gemini requested tool: {function_call.name}")
        print(f"Arguments intercepted: {function_call.args}\n")

        # STEP 5: Execute our local code using Gemini's extracted arguments
        if function_call.name == "get_current_weather":
            # Using dictionary unpacking (**kwargs) to safely feed the args
            tool_output = get_current_weather(**function_call.args)
            print(f"Executed local function. Output: {tool_output}\n")

            # STEP 6: Send the tool result back into the chat session
            for attempt in Retrying(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2)):
                with attempt:
                    final_response = chat.send_message(
                        types.Part.from_function_response(
                            name=function_call.name,
                            response={"content": tool_output}
                        )
                    )
            
            print("--- Gemini's Final Answer ---")
            print(final_response.text)
            print("-----------------------------")
    else:
        # If Gemini didn't need a tool, it just gives standard text back
        print(response.text)

if __name__ == "__main__":
    main()