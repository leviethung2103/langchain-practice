# Run uvicorn main:app --reload

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
import string
import json
import random
from fastapi.responses import StreamingResponse
from typing import Any, List, Optional
import asyncio
from langchain.schema import HumanMessage
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from typing import Any, List
from loguru import logger
import sys

app = FastAPI()
load_dotenv(override=True)

BEAR_TOKEN_API_KEY = os.getenv("BEAR_TOKEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOGURU_LEVEL = os.getenv("LOGURU_LEVEL", "INFO")


logger.remove()  # for someone not familiar with the lib, whats going on here?
logger.add(sys.stdout, level=LOGURU_LEVEL)

logger.debug("openAPI:", OPENAI_API_KEY)


def generate_unique_id():
    timestamp = int(time.time())
    random_string = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{timestamp}{random_string}"


# Define the input model for the API endpoint
class ChatCompletionInput(BaseModel):
    """
    Ref: https://docs.mistral.ai/api/#operation/createChatCompletion
    """

    model: str
    messages: list[dict[str, str]]
    stream: bool = False
    temperature: float = 0.7
    top_p: int = 1
    safe_prompt: bool = False
    max_tokens: int = None
    random_seed: int = None


class ChatCompletionMessage(BaseModel):
    role: str
    content: str
    tool_calls: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str
    logprobs: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


# Streaming
class ChatCompletionChunkMessage(BaseModel):
    content: str
    role: Optional[str]
    tool_calls: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[int] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ChatCompletionStreamingResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# Configure CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def split_method(content, method="parapgrah"):
    import re

    if method == "paragraph":
        paragraphs = content.split("\n")
        return paragraphs
    elif method == "sentence":
        sentences = re.split(r"[.!?]+", content)
        return sentences
    elif method == "word":
        words = content.split()
        chunk_size = 50
        chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks


async def fake_generator():
    dummy_content = "The \"best\" French cheese is subjective as it depends on personal preferences. However, some popular and highly regarded French cheeses include:\n\n1. Brie de Meaux - A soft, creamy cheese with a mild, slightly sweet and nutty flavor.\n2. Camembert de Normandie - Similar to Brie, but with a stronger, more earthy flavor.\n3. Roquefort - A blue cheese made from sheep's milk, known for its tangy, salty, and slightly sweet flavor.\n4. Comté - A firm, nutty cheese made from unpasteurized cow's milk.\n5. Reblochon - A soft, washed-rind cheese with a creamy, slightly pungent flavor, often used in the traditional French dish, tartiflette.\n6. Époisses - A strong, soft cheese with a pungent aroma and a creamy, slightly salty flavor.\n7. Gruyère - A firm, nutty cheese often used in cooking, such as in French onion soup.\n8. Chèvre - A goat cheese that can vary widely in flavor and texture, from mild and creamy to strong and crumbly.\n\nThese are just a few examples of the many delicious French cheeses available. It's worth trying several to find your favorite!"
    msg_id = generate_unique_id()

    chunks = split_method(dummy_content, method="sentence")

    created_at = int(time.time())

    for i, chunk in enumerate(chunks):
        response_chunk = ChatCompletionChunk(
            id=msg_id,
            object="chat.completion.chunk",
            created=created_at,
            model="mistral-medium-latest",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkMessage(content=chunk, role="assistant"),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
        )
        data = json.dumps(response_chunk.dict())
        msg = f"data: {data}\n\n"
        await asyncio.sleep(0.1)  # Simulate streaming delay
        yield msg

    # add the last chunk with finish reason
    last_chunk = ChatCompletionChunk(
        id=msg_id,
        object="chat.completion.chunk",
        created=created_at,
        model="mistral-medium-latest",
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkMessage(content="", role="assistant"),
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )

    last_data = json.dumps(last_chunk.dict())
    last_msg = f"data: {last_data}\n\n"
    yield last_msg

    # add the last chunk [DONE]
    last_msg = f"data: [DONE]\n\n"
    yield last_msg


class CustomAsyncCallbackHandler(AsyncIteratorCallbackHandler):
    """Tutorial: https://www.youtube.com/watch?v=y2cRcOPHL_U"""

    content: str = ""
    final_answer: bool = False
    msg_id = generate_unique_id()
    created_at = int(time.time())

    async def on_llm_new_token(self, token: str, **kwargs: Any):
        self.content += token
        if token is not None and token != "":
            chunk = ChatCompletionChunk(
                id=self.msg_id,
                object="chat.completion.chunk",
                created=self.created_at,
                model="mistral-medium-latest",
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkMessage(content=token, role="assistant"),
                        finish_reason=None,
                        logprobs=None,
                    )
                ],
            )

            chunk_dict = json.dumps(chunk.dict())
            new_token = f"data: {chunk_dict}\n\n"
            self.queue.put_nowait(new_token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # ! BUG cannot put the message into the queue here
        # # add the last chunk with finish reason
        # last_chunk = ChatCompletionChunk(
        #     id=self.msg_id,
        #     object="chat.completion.chunk",
        #     created=self.created_at,
        #     model="mistral-medium-latest",
        #     choices=[
        #         ChatCompletionChunkChoice(
        #             index=0,
        #             delta=ChatCompletionChunkMessage(content="", role="assistant"),
        #             finish_reason="stop",
        #             logprobs=None,
        #         )
        #     ],
        # )

        # last_data = json.dumps(last_chunk.dict())
        # last_msg = f"data: {last_data}\n\n"
        # # This ensures that the queue is updated before the done event is set.
        # await self.queue.put(last_msg)

        # # add the last chunk [DONE]
        # last_msg = f"data: [DONE]\n\n"
        # # This ensures that the queue is updated before the done event is set.
        # await self.queue.put(last_msg)

        # self.final_answer = True  # Only add the last chunk once
        logger.debug(f"Response: {self.content}")
        self.done.set()


async def langchain_generator(input_text: str | list):
    # callback = AsyncIteratorCallbackHandler()
    callback = CustomAsyncCallbackHandler()
    llm = ChatOpenAI(streaming=True, verbose=True, callbacks=[callback], api_key=OPENAI_API_KEY)

    if isinstance(input_text, str):
        task = asyncio.create_task(llm.agenerate(messages=[[HumanMessage(content=input_text)]]))
    elif isinstance(input_text, list):
        logger.debug(f"converted input: {convert_to_messages(input_text)}")
        task = asyncio.create_task(llm.agenerate(messages=convert_to_messages(input_text)))

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task


def convert_to_messages(message_list) -> List[List[BaseMessage]]:
    """
    Convert a list of dictionaries into a list of HumanMessage and AIMessage objects.

    Args:
        message_list (list): A list of dictionaries, where each dictionary has a 'role' and 'content' key.

    Returns:
        list: A list of HumanMessage and AIMessage objects.
    """
    messages = []
    for msg in message_list:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "system":
            messages.append(AIMessage(content=msg["content"]))
    # Return the list of list of messages
    return [messages]


# Define the API endpoint
@app.post("/v0/chat/completions")
async def chat_completions(request: Request, data: ChatCompletionInput):
    """Fake General AI chat completion endpoint"""
    # Check if the Bearer Token is provided in the request headers
    if "Authorization" not in request.headers:
        raise HTTPException(status_code=401, detail="Bearer Token not provided")

    # Extract the Bearer Token from the request headers
    bear_token = request.headers["Authorization"].split(" ")[1]

    # Check if the Bearer Token is valid
    if bear_token != BEAR_TOKEN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid Bearer Token")

    # Process the request and return the response
    dummy_content = "The \"best\" French cheese is subjective as it depends on personal preferences. However, some popular and highly regarded French cheeses include:\n\n1. Brie de Meaux - A soft, creamy cheese with a mild, slightly sweet and nutty flavor.\n2. Camembert de Normandie - Similar to Brie, but with a stronger, more earthy flavor.\n3. Roquefort - A blue cheese made from sheep's milk, known for its tangy, salty, and slightly sweet flavor.\n4. Comté - A firm, nutty cheese made from unpasteurized cow's milk.\n5. Reblochon - A soft, washed-rind cheese with a creamy, slightly pungent flavor, often used in the traditional French dish, tartiflette.\n6. Époisses - A strong, soft cheese with a pungent aroma and a creamy, slightly salty flavor.\n7. Gruyère - A firm, nutty cheese often used in cooking, such as in French onion soup.\n8. Chèvre - A goat cheese that can vary widely in flavor and texture, from mild and creamy to strong and crumbly.\n\nThese are just a few examples of the many delicious French cheeses available. It's worth trying several to find your favorite!"
    created_at = int(time.time())

    if not data.stream:
        response = ChatCompletionResponse(
            id=generate_unique_id(),
            object="chat.completion",
            created=created_at,
            model="mistral-medium-latest",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=dummy_content, tool_calls=None),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=ChatCompletionUsage(prompt_tokens=16, total_tokens=328, completion_tokens=312),
        )
        return response
    else:
        generator = fake_generator()
        return StreamingResponse(generator, media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, data: ChatCompletionInput):
    """Langchain Generator chat completion endpoint using Langchain"""
    # Check if the Bearer Token is provided in the request headers
    if "Authorization" not in request.headers:
        raise HTTPException(status_code=401, detail="Bearer Token not provided")

    # Extract the Bearer Token from the request headers
    bear_token = request.headers["Authorization"].split(" ")[1]

    # Check if the Bearer Token is valid
    if bear_token != BEAR_TOKEN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid Bearer Token")

    # Process the request and return the response
    dummy_content = "The \"best\" French cheese is subjective as it depends on personal preferences. However, some popular and highly regarded French cheeses include:\n\n1. Brie de Meaux - A soft, creamy cheese with a mild, slightly sweet and nutty flavor.\n2. Camembert de Normandie - Similar to Brie, but with a stronger, more earthy flavor.\n3. Roquefort - A blue cheese made from sheep's milk, known for its tangy, salty, and slightly sweet flavor.\n4. Comté - A firm, nutty cheese made from unpasteurized cow's milk.\n5. Reblochon - A soft, washed-rind cheese with a creamy, slightly pungent flavor, often used in the traditional French dish, tartiflette.\n6. Époisses - A strong, soft cheese with a pungent aroma and a creamy, slightly salty flavor.\n7. Gruyère - A firm, nutty cheese often used in cooking, such as in French onion soup.\n8. Chèvre - A goat cheese that can vary widely in flavor and texture, from mild and creamy to strong and crumbly.\n\nThese are just a few examples of the many delicious French cheeses available. It's worth trying several to find your favorite!"
    created_at = int(time.time())

    if not data.stream:
        response = ChatCompletionResponse(
            id=generate_unique_id(),
            object="chat.completion",
            created=created_at,
            model="mistral-medium-latest",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=dummy_content, tool_calls=None),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=ChatCompletionUsage(prompt_tokens=16, total_tokens=328, completion_tokens=312),
        )
        return response
    else:
        # generator = langchain_generator(input_text)
        # return StreamingResponse(generator, media_type="text/event-stream")

        # data:messages is a list of dictionaries, convert it to a list of HumanMessage and AIMessage objects
        response = langchain_generator(data.messages)
        return StreamingResponse(response, media_type="text/event-stream")
