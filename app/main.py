import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from gpt.CGIPrivateGPT import run_model, execute

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


@api.get("/run_model/")
async def initiate_model():
    run_status = run_model()
    return {"Status": run_status}


@api.get("/pvt_gpt/")
async def pvt_gpt_response(query: str):
    # Your chatbot logic here
    response = pvt_gpt_generate_response(query)
    return {"reply": response}


def pvt_gpt_generate_response(query: str):
    # Add your chatbot logic to generate a response based on the query
    # Here's a simple example that echoes the query as the response
    response = execute(query)
    return response


if __name__ == "__main__":
    uvicorn.run(app="app.main:api",
                host="0.0.0.0",
                port=5000,
                reload=True,
                workers=1
                )
