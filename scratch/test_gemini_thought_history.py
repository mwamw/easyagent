import asyncio
import os
from google import genai
from google.genai import types

async def main():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-3.1-pro"
    
    # Send a message that returns a thought
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thought=True)
    )
    
    print("Sending first msg...")
    response = await client.aio.models.generate_content(
        model=model,
        contents="What is 2+2? Think step by step.",
        config=config,
    )
    
    print("Response parts:")
    for p in response.candidates[0].content.parts:
        print(type(p), p)
        
    print("Sending second msg...")
    history = [
        types.Content(role="user", parts=[types.Part.from_text("What is 2+2? Think step by step.")]),
        response.candidates[0].content,
        types.Content(role="user", parts=[types.Part.from_text("Now what is 3+3?")])
    ]
    try:
        response2 = await client.aio.models.generate_content(
            model=model,
            contents=history,
            config=config,
        )
        print("Success on 2nd msg")
    except Exception as e:
        print("Error on 2nd msg:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
