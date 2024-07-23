from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0
)
# print(chat([HumanMessage(content="Write me a song about sparkling water.")]))

# invoke: Must a PromptValue, str, or list of BaseMessages
# print(chat.invoke("Write me a song about sparkling water."))

from langchain_core.messages import HumanMessage, SystemMessage

# messages = [
#     SystemMessage(
#         content="You are a helpful assistant! Your name is Bob."
#     ),
#     HumanMessage(
#         content="What is your name?"
#     )
# ]

messages = [HumanMessage(content="What is your name?")]


print(chat.invoke(messages))