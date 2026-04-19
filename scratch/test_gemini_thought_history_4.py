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
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig()
    )
    
    response = await client.aio.models.generate_content(
        model=model,
        contents="What is 2+2? Give me thinking block.",
        config=config,
    )
    
    content = response.candidates[0].content
    print("Parts dumped from API response:")
    for p in content.parts:
        print(p.model_dump())
        
if __name__ == "__main__":
    asyncio.run(main())
