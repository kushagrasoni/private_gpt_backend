from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# from app.gpt.CGIPrivateGPT import run_model, execute
from app.gpt.gpt_improved_code import run_model, execute


api = FastAPI(timeout=1200)

# Define allowed origins
# origins = [
#     "http://localhost",
#     "http://0.0.0.0",
#     "http://localhost:3000",
#     "http://0.0.0.0:3000",
#     "http://54.167.71.250:3000"
#     "http://54.167.71.250",
#     "http://ec2-54-167-71-250.compute-1.amazonaws.com",
#     "http://172.31.21.244:3000"
#     # Add more allowed origins as needed
# ]

origins = ["*"]

# Configure CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# global knowledge_base, system_prompt, tokenizer, model_4bit, device
global knowledge_base, docs, embeddings, system_prompt

@api.get("/")
async def main():
    return {"Status": "OK"}


@api.get("/run_model")
async def initiate_model():
    # global knowledge_base, system_prompt, tokenizer, model_4bit, device
    # knowledge_base, system_prompt, tokenizer, model_4bit, device = run_model()
    global knowledge_base, docs, embeddings, system_prompt
    knowledge_base, docs, embeddings, system_prompt = run_model()
    return {"Status": True}


@api.get("/pvt_gpt")
async def pvt_gpt_response(query: str):
    start_time = datetime.now()
    response = pvt_gpt_generate_response(query)
    end_time = datetime.now()
    time_taken = end_time - start_time

    print('Response Time: ',time_taken)
    return {"reply": response}


@api.get("/pvt_gpt2")
async def pvt_gpt_response(query: str):
    # Your chatbot logic here
    #response = pvt_gpt_generate_response(query)
    return {"reply": query}


def pvt_gpt_generate_response(query: str):
    # global knowledge_base, system_prompt, tokenizer, model_4bit, device
    # Add your chatbot logic to generate a response based on the query
    # Here's a simple example that echoes the query as the response

    # response = execute(query, knowledge_base, system_prompt, tokenizer, model_4bit, device)

    global knowledge_base, docs, embeddings, system_prompt
    response = execute(query, knowledge_base, docs, embeddings, system_prompt)

    return response


if __name__ == "__main__":
    uvicorn.run(app="app.main:api",
                host="0.0.0.0",
                port=5000,
                reload=True,
                workers=1
                )
