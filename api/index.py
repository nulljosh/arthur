from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str = "fn "
    max_tokens: int = 50
    temperature: float = 0.8

@app.get("/")
def root():
    return {"status": "ok", "message": "aether v1.0 API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "name": "aether",
        "version": "1.0.0",
        "params": "3.5M",
        "github": "https://github.com/nulljosh/aether"
    }

@app.post("/generate")
def generate(request: GenerateRequest):
    return {
        "prompt": request.prompt,
        "output": request.prompt + " [aether v1.0 inference]",
        "tokens_generated": 10
    }
