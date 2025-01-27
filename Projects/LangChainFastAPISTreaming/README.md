# OpenAI GPT-3.5-turbo Streaming API with FastAPI 

This project demonstrates how to create a real-time conversational AI by streaming responses from OpenAI's GPT-3.5-turbo model. It uses FastAPI to create a web server that accepts user inputs and streams generated responses back to the user.

## Running the Project

1. Clone the repository.
2. Install Python (Python 3.7+ is recommended).
3. Update the value of OpenAI API key in .env.example and rename file into .env
4. Install necessary libraries. This project uses FastAPI, uvicorn, LangChain, among others. You can install them with pip: `pip install fastapi uvicorn langchain`.
5. Add your OpenAI API key to the `.env` file.
6. Start the FastAPI server by running `uvicorn main:app --port 7000` in the terminal.
7. Access the application by opening your web browser and navigating to `localhost:7000`.

In the `index.html` file, you need to update the http into 127.0.0.1:7000 instead of localhost:7000 -> CORS problem

Serving the html file
```bash
python3 -m http.server 8888
```

Access the web URL via: http://127.0.0.1:8888

Note: Ensure the appropriate CORS settings if you're not serving the frontend and the API from the same origin.


## Web
1. Define the async function sendMessage()
2. reader.read().then(function processResult(**result**) { ... }):
This line initiates the reading of data from a stream using the reader.read() method.
The then() method is used to handle the asynchronous result of the read() operation.
The processResult function is the callback function that will be executed when the read() operation completes.
3. if (result.done) return;:
This checks if the stream has finished reading all the data, indicated by the result.done property being true.
If the stream has finished reading, the function simply returns, effectively terminating the current iteration of the loop.
4. The recursive call ensures that the function continues to read and process data from the stream until it is completely read.

## Project Overview

The project uses an HTML interface for user input. The user's input is sent to a FastAPI server, which forwards it to the GPT-3.5-turbo model. The generated response is streamed back to the user, simulating a real-time conversation. 
