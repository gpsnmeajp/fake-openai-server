from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Union
from sentence_transformers import SentenceTransformer
from base64 import b64encode

import logging
import logging.config
from logging import Logger
from LoggerConfig import LoggerConfig

logging.config.dictConfig(
    LoggerConfig.generate(log_file=None, stdout=True),
)
logger: Logger = logging.getLogger("embedding-api-server")
logger.setLevel(logging.INFO)

app = FastAPI()
model = SentenceTransformer("cl-nagoya/ruri-large")


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str]]
    encoding_format: Literal["float", "base64"] | None = "float"


@app.post("/v1/embeddings")
def create_embedding(request: EmbeddingRequest):
    logger.info(f"Received request: {request}")
    texts = [request.input] if isinstance(request.input, str) else request.input
    embeddings = model.encode(texts, convert_to_numpy=True)
    data = []
    for idx, emb in enumerate(embeddings):
        # Match OpenAI's encoding_format contract when callers ask for base64 blobs.
        emb_payload = (
            b64encode(emb.astype("float32").tobytes()).decode("ascii")
            if request.encoding_format == "base64"
            else emb.tolist()
        )
        data.append({"object": "embedding", "embedding": emb_payload, "index": idx})
    res = {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    logger.debug(f"Response: {res}")
    return res
