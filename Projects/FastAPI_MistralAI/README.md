## FastAPI & Mistral AI

Cloning the Mistral AI using the FastAPI
Mistral AI Documentation: https://docs.mistral.ai/api/#operation/createChatCompletion

In this project, we will clone the Mistral AI Endpoints


## Project Overview
![alt text](images/system.png)

## Endpoints

`/v0/chat/completions`
- Support no streaming response
- Support FAKE streaming response

`/v1/chat/completions`
- Support no streaming response
- Support LANGCHAIN streaming response


### Usage
```bash
# Run server
cp -r .env.example .env
# update the variables in .env file
uvicorn main:app --reload
```

Create .env file and set the BEAR_TOKEN_API_KEY=<token-value>

**Option 1: Use Postman**
![alt text](images/authentication.png)
![alt text](images/postman.png)

**Option 2: Use `test_stream.py` file**
```bash
python test_stream.py`
```

### Features
1. Return the response without streaming 
No streaming response

1. Return the streaming response from fake generator
Streaming response with FastAPI

1. Return the streaming from Langchain
Streaming response with FastAPI + Langchain


### Limitation
- Cannot add the message "DONE" at the end of stream