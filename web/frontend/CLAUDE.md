# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **React frontend** for the NJ Quant quantitative trading system. It provides a web UI for signal management, position tracking, data browsing, and cron job monitoring. The backend is a Python FastAPI server (located at `../server/`).

## Running the Frontend

```bash
# Install dependencies
npm install

# Start dev server (port 3000, proxies /api to localhost:8080)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Architecture

```
src/
├── App.tsx              # Main layout with routing and navigation menu
├── main.tsx             # React entry point
└── pages/
    ├── SignalTable/      # Trading signal看板 — confirm/skip signals
    ├── Positions/       # 持仓管理 — open positions, history, capital logs
    ├── DataBrowser/      # 数据浏览 — stock data search and detail view
    └── CronTracker/      # 定时任务 — cron job status and execution logs
```

**Tech Stack**: React 18 + TypeScript + Vite + Ant Design 5 + React Router 6 + Axios + dayjs

## API Integration

All API calls go through `/api/*` and are proxied to `http://localhost:8080` (see `vite.config.ts`). No base URL configuration needed — all pages use relative paths with axios.

Key API endpoints used:
- `GET /api/signals` — signal list with pagination
- `POST /api/signals/:id/confirm` — confirm a signal
- `POST /api/signals/:id/skip` — skip a signal
- `GET /api/positions/overview` — portfolio summary
- `GET /api/positions/` — positions (query `status=OPEN` or `status=CLOSED`)
- `GET /api/data/stocks` — stock data with search/pagination
- `GET /api/cron/` — cron execution logs
- `GET /api/cron/status` — data completeness status

## Adding a New Page

1. Create `src/pages/{PageName}/index.tsx` with a React component
2. Add the route in `App.tsx`: `<Route path="/{kebab-case}" element={<PageName />} />`
3. Add the menu item in `items` array with an icon from `@ant-design/icons`
