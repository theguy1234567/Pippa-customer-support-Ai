from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent.agent import SupportAgent
from .api.routes import router

@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.agent = SupportAgent()
    yield


app = FastAPI(title="Hiver Support Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "hiver-support-agent", "status": "ready"}




