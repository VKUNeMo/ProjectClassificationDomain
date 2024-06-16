import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(1, os.getenv('PATH_ROOT'))
import torch
import logging
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from processing_data.getLexicalFeature import getLexicalInputNN
from domain_feature.lexical_feature import LexicalURLFeature
from api.model import load_model, predict, NN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DomainRequest(BaseModel):
    domain: str


class DomainResponse(BaseModel):
    domain: str
    entropy: float
    percentageDigits: float
    domainLength: int
    specialChars: int
    result: int


@app.post("/api/predict", response_model=DomainResponse)
async def get_predict(request: DomainRequest):
    try:
        logger.info(f"Received request: {request}")
        domain = request.domain
        model_nn = load_model()
        lexical_input = torch.tensor(getLexicalInputNN(domain), dtype=torch.float)
        lexical_input = lexical_input.unsqueeze(0)
        logger.info(f"Lexical input: {lexical_input}")
        lexical = LexicalURLFeature(domain)

        prediction = predict(model_nn, lexical_input).item()

        response = DomainResponse(
            domain=domain,
            entropy=lexical.get_entropy(),
            percentageDigits=lexical.get_percentage_digits(),
            domainLength=lexical.get_length_to_feed_model(),
            specialChars=lexical.get_count_special_characters(),
            result=prediction
        )
        return JSONResponse(
            status_code=200,
            content=response.dict()
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error", "details": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
