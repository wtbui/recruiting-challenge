from fastapi import FastAPI
from app.routers import profile_router
import uvicorn

tags_metadata = [
    {
        "name": "profile",
        "description": "Operations with profile"
    }
]

app = FastAPI(openapi_tags=tags_metadata)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

# Root Endpoint
@app.get("/")
async def root():
    return {"William Bui - IndentifAI Recruiting Challenge"}

# Routers
app.include_router(profile_router)