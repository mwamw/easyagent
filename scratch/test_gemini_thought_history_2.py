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
    # Actually wait, maybe our EasyAgent code formats history with `thought_signature` or `thought=True` and puts them into `types.Part` dynamically in _api_client.py, but they are not valid in the model response object when we pass them back?
    
    # Try creating custom parts just like GoogleNativeCodec does in EasyAgent
    history = [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"thought": True, "text": "thinking...", "thought_signature": "signature"}, {"text": "hi"}]},
        {"role": "user", "parts": [{"text": "again"}]}
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
        print(repr(e))

if __name__ == "__main__":
    asyncio.run(main())
