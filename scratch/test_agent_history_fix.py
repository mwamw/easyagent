import asyncio
import os
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
    
    # Send a request to get a thought
    history1 = [
        {"role": "user", "parts": [{"text": "What is 2+2? Give me thinking block."}]}
    ]
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=history1,
            config=config,
        )
    except Exception as e:
        print("Initial call failed:")
        print(e)
        return

    content = response.candidates[0].content
    parts_dicts = []
    for p in content.parts:
        pd = {}
        if p.text: pd["text"] = p.text
        if p.thought: pd["thought"] = p.thought
        if p.thought_signature: pd["thought_signature"] = p.thought_signature
        parts_dicts.append(pd)
        
    print("Received parts:")
    for pd in parts_dicts:
        print(pd)
        
    # Now try sending back exactly what we got
    history2 = history1 + [
        {"role": "model", "parts": parts_dicts},
        {"role": "user", "parts": [{"text": "Now add 3"}]}
    ]
    
    print("\nSending back WITH thought_signature:")
    try:
        await client.aio.models.generate_content(
            model=model, contents=history2, config=config
        )
        print("Success!")
    except Exception as e:
        print(repr(e))
        
    # Now try WITHOUT thought_signature
    for pd in parts_dicts:
        pd.pop("thought_signature", None)
        
    history3 = history1 + [
        {"role": "model", "parts": parts_dicts},
        {"role": "user", "parts": [{"text": "Now add 3"}]}
    ]
    
    print("\nSending back WITHOUT thought_signature:")
    try:
        await client.aio.models.generate_content(
            model=model, contents=history3, config=config
        )
        print("Success without thought_signature!")
    except Exception as e:
        print(repr(e))

if __name__ == "__main__":
    asyncio.run(main())
