import os
import openai
import dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from query_vector_db import retrieve  # Assuming retrieve is in query_vector_db.py

from fastapi.middleware.cors import CORSMiddleware




dotenv.load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Adjust the port if different
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define a Pydantic model for the request body
class QueryRequest(BaseModel):
    query: str


# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    # Call the retrieve function with the query from the request
    contexts, metadata = retrieve(request.query, openai_client)
    return {"contexts": contexts, "metadata": metadata}
