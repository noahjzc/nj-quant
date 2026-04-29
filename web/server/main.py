from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime

from web.server.api import signals, positions, data_browser, cron_status

app = FastAPI(title="NJ Quant Signal Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(positions.router, prefix="/api/positions", tags=["positions"])
app.include_router(data_browser.router, prefix="/api/data", tags=["data"])
app.include_router(cron_status.router, prefix="/api/cron", tags=["cron"])


@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}
