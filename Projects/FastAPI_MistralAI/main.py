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
from typing import Optional
import asyncio

app = FastAPI()
load_dotenv()

BEAR_TOKEN_API_KEY = os.getenv("BEAR_TOKEN_API_KEY")


def generate_unique_id():
    timestamp = int(time.time())
    random_string = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    unique_id = f"{timestamp}{random_string}"
    return unique_id


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


async def generate_response():
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


# Define the API endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, data: ChatCompletionInput):
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
        generator = generate_response()
        return StreamingResponse(generator, media_type="text/event-stream")
