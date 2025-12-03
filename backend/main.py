from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import v1

app = FastAPI(
    title="MoonBoard Grade Guesser API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1.router, prefix="/v1")


@app.get("/")
def root() -> dict:
    return {"message": "MoonBoard Grade Guesser API", "docs": "/docs"}
