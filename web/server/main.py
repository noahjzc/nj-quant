from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime

from web.server.api import signals, positions, data_browser, cron_status

app = FastAPI(
    title="NJ Quant Signal Dashboard",
    version="1.0.0",
    description="A股每日实盘信号管线 — 信号生成、持仓管理、数据浏览、任务追踪",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router, prefix="/signals", tags=["signals"])
app.include_router(positions.router, prefix="/positions", tags=["positions"])
app.include_router(data_browser.router, prefix="/data", tags=["data"])
app.include_router(cron_status.router, prefix="/cron", tags=["cron"])


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}
