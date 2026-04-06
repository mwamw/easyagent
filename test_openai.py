from pprint import pprint
import pydantic
from openai.types.chat.chat_completion_message import ChatCompletionMessage

print("ChatCompletionMessage fields:")
pprint(list(ChatCompletionMessage.model_fields.keys()))

