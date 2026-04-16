from google import genai
client = genai.Client(api_key="123", http_options={'base_url': 'http://foo'})
print("Success! base_url:", client._api_client.base_url if hasattr(client, '_api_client') else client)
