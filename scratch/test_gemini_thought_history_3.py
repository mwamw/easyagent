import asyncio
import os
import json
from google import genai
from google.genai import types

async def main():
    api_key = "sk-b5685adb3LBGrqpMpzyJpwWrlKQgli1Sg1TLvwxBYUZKkk5c"
    client = genai.Client(
        http_options={'base_url': 'http://210.45.70.84:30000'},
        api_key=api_key
    )
    model = "gemini-3-flash"
    config = types.GenerateContentConfig()
    
    print("1...")
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents="What is 2+2?",
            config=config,
        )
    except Exception as e:
        print("Initial call failed:")
        print(e)
        return

    print("Got response")
    
    history = [
        types.Content(role="user", parts=[types.Part.from_text(text="What is 2+2?")]),
        response.candidates[0].content,
        types.Content(role="user", parts=[types.Part.from_text(text="Now what is 3+3?")])
    ]
    
    print("2...")
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

if __name__ == "__main__":
    asyncio.run(main())
