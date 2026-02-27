from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="aether", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8

class GenerateResponse(BaseModel):
    prompt: str
    output: str
    tokens_generated: int

@app.get("/")
def root():
    return {
        "name": "aether",
        "version": "1.0.0",
        "description": "Nano transformer LLM, 3.5M params",
        "github": "https://github.com/nulljosh/aether"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "name": "aether",
        "version": "1.0.0",
        "params": "3.5M",
        "layers": 12,
        "embed_dim": 256,
        "final_loss": 0.09,
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    # Mock response for now
    output = request.prompt + " [inference engine running on M4 locally]"
    return GenerateResponse(
        prompt=request.prompt,
        output=output,
        tokens_generated=10
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
