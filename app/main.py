from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.gpt.CGIPrivateGPT import run_model, execute

api = FastAPI()

# Define allowed origins
origins = [
    "http://localhost",
    "http://localhost:3000",
    # Add more allowed origins as needed
]

# Configure CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


global knowledge_base, system_prompt, tokenizer, model_4bit, device

@api.get("/")
async def main():
    return {"Status": "OK"}


@api.get("/run_model")
async def initiate_model():
    global knowledge_base, system_prompt, tokenizer, model_4bit, device
    knowledge_base, system_prompt, tokenizer, model_4bit, device = run_model()
    return {"Status": True}


@api.get("/pvt_gpt")
async def pvt_gpt_response(query: str):
    # Your chatbot logic here
    response = pvt_gpt_generate_response(query)
    return {"reply": response}


def pvt_gpt_generate_response(query: str):
    global knowledge_base, system_prompt, tokenizer, model_4bit, device
    # Add your chatbot logic to generate a response based on the query
    # Here's a simple example that echoes the query as the response
    response = execute(query, knowledge_base, system_prompt, tokenizer, model_4bit, device)
    return response


if __name__ == "__main__":
    uvicorn.run(app="app.main:api",
                host="0.0.0.0",
                port=5000,
                reload=True,
                workers=1
                )
