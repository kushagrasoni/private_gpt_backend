import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, status
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
import spacy

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



# Load the spaCy language model
nlp = spacy.load('en_core_web_sm')


# Create a new ChatBot instance
chatbot = ChatBot('My ChatBot')

# # Create a new trainer for the ChatBot
# trainer = ListTrainer(chatbot)
#
# # Define a list of conversation examples to train the chatbot
# conversation_examples = [
#     'Hello',
#     'Hi there!',
#     'How are you?',
#     'I am good.',
#     'That is good to hear.',
#     'Thank you',
#     'You\'re welcome.'
# ]
#
# # Train the ChatBot on the list of conversation examples
# trainer.train(conversation_examples)


@api.get("/train_chatbot")
async def train_chatbot():
    # Create a new trainer for the ChatBot
    trainer = ChatterBotCorpusTrainer(chatbot)

    # Train the ChatBot on a corpus of data
    trainer.train('chatterbot.corpus.english')

    return "Model Trained"


@api.get("/pvt_gpt")
async def pvt_gpt_response(query: str):
    response = chatbot.get_response(query)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to generate response"
        )
    return {"reply": response.text}


# Other code...


if __name__ == "__main__":
    uvicorn.run(app="app.main:api",
                host="0.0.0.0",
                port=5000,
                reload=True,
                workers=1
                )
