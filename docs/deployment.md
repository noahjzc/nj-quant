# 实盘信号管线部署指南

## 目录

- [本地验证](#本地验证)
- [服务器部署](#服务器部署)
- [每日运维](#每日运维)
- [目录结构](#目录结构)

---

## 本地验证

### 1. 安装依赖

```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
# 或
source .venv/bin/activate  # Linux/Mac

# 安装新增依赖
pip install tushare akshare fastapi uvicorn
```

### 2. 初始化数据库（首次）

本地已有 PostgreSQL 和 `quant_db` 数据库可跳过。

```bash
# 创建数据库和用户（用 postgres 超级用户）
sudo -u postgres psql -f back_testing/data/scripts/setup_database.sql

# 创建表结构
psql -U quant_user -d quant_db -f scripts/server_setup_02_tables.sql

# 创建实盘信号管线专属表（4张新表）
psql -U quant_user -d quant_db -f signal_pipeline/schema.sql
```

> **注意**：数据库名为 `quant_db`，用户为 `quant_user`，不是 `njquant`。

### 3. 配置环境变量

```bash
# 创建 .env 文件
cp deploy/.env.example .env

# 编辑 .env，填入实际值
nano .env
```

`.env` 内容：

```bash
TUSHARE_TOKEN=你的Tushare_TOKEN
CACHE_DIR=cache/daily_rotation
DB_HOST=localhost
DB_PORT=5432
DB_NAME=quant_db
DB_USER=quant_user
DB_PASSWORD=你的数据库密码
```

> Tushare Token 在 [tushare.pro](https://tushare.pro) 注册后获取，2000 积分档足够每晚 3-5 次调用。

### 4. 启动后端 API

```bash
# 激活虚拟环境后
cd D:\workspace\code\mine\quant\nj-quant  # 项目根目录
uvicorn web.server.main:app --host 127.0.0.1 --port 8080 --reload
```

验证：

```bash
curl http://127.0.0.1:8080/health
# 期望返回: {"status":"ok"}
```

### 5. 启动前端

```bash
cd web/frontend
npm install
npm run dev
```

访问 http://localhost:3000 应能看到看板界面。

### 6. 测试信号脚本（可选）

```bash
# 14:25 的信号脚本可在任意时间手动触发（用于验证）
python signal_pipeline/intraday_signal.py
```

---

## 服务器部署

### 前提条件

- Linux 服务器（Ubuntu 20.04+ 或 CentOS 8+）
- PostgreSQL 14+
- Python 3.10+
- Node.js 18+（构建 React 前端用）
- Nginx
- systemctl（systemd）
- Tushare Pro Token

### 步骤 1：创建系统用户

```bash
# 创建专用用户（可选，也可以用现有用户）
sudo useradd -m -s /bin/bash njquant
sudo mkdir -p /home/njquant
sudo chown njquant:njquant /home/njquant
```

### 步骤 2：上传代码

```bash
# 在本地打包
git archive --format=tar HEAD > nj-quant.tar

# 上传到服务器（用 scp 或其他方式）
scp nj-quant.tar njquant@your-server:/home/njquant/

# 在服务器解压
ssh njquant@your-server
cd /home/njquant
tar -xf nj-quant.tar
```

### 步骤 3：安装系统依赖

```bash
sudo apt update
sudo apt install -y python3.10-venv python3-pip nodejs npm nginx postgresql systemctl
```

### 步骤 4：初始化数据库

```bash
# 用 postgres 超级用户执行
sudo -u postgres psql -f /home/njquant/nj-quant/scripts/server_setup_01_init.sql
sudo -u postgres psql -U quant_user -d quant_db -f /home/njquant/nj-quant/scripts/server_setup_02_tables.sql
sudo -u postgres psql -U quant_user -d quant_db -f /home/njquant/nj-quant/signal_pipeline/schema.sql
```

> `server_setup_01_init.sql` 中的密码是自动生成的占位密码，实际部署时应修改为安全的随机密码。

### 步骤 5：配置 Python 虚拟环境

```bash
cd /home/njquant/nj-quant
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 步骤 6：配置环境变量

```bash
cp /home/njquant/nj-quant/deploy/.env.example /home/njquant/nj-quant/.env
nano /home/njquant/nj-quant/.env
```

```bash
TUSHARE_TOKEN=你的Tushare_TOKEN
CACHE_DIR=/home/njquant/nj-quant/cache/daily_rotation
DB_HOST=localhost
DB_PORT=5432
DB_NAME=quant_db
DB_USER=quant_user
DB_PASSWORD=你的数据库密码
```

### 步骤 7：构建 React 前端

```bash
cd /home/njquant/nj-quant/web/frontend
npm install
npm run build
```

构建产物在 `dist/` 目录。

### 步骤 8：创建日志目录

```bash
mkdir -p /home/njquant/nj-quant/logs
mkdir -p /home/njquant/nj-quant/cache/daily_rotation/daily
```

### 步骤 9：安装 systemd 服务

```bash
sudo cp /home/njquant/nj-quant/deploy/nj-quant-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nj-quant-web
sudo systemctl start nj-quant-web

# 验证
sudo systemctl status nj-quant-web
```

### 步骤 10：配置 Nginx

```bash
sudo cp /home/njquant/nj-quant/deploy/nginx.conf /etc/nginx/sites-available/nj-quant
sudo ln -s /etc/nginx/sites-available/nj-quant /etc/nginx/sites-enabled/
sudo nginx -t  # 检查配置语法
sudo systemctl reload nginx
```

如果服务器有防火墙：

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp  # 如果后续配置 HTTPS
```

### 步骤 11：配置 Cron 任务

```bash
# 编辑 crontab
crontab -e

# 粘贴以下内容（来自 deploy/crontab.txt）
TUSHARE_TOKEN=你的Tushare_TOKEN

# 14:25 — 盘中信号生成（周一至周五）
25 14 * * 1-5  cd /home/njquant/nj-quant && .venv/bin/python signal_pipeline/intraday_signal.py >> logs/intraday.log 2>&1

# 18:00 — 盘后数据补全（周一至周五）
0 18 * * 1-5  cd /home/njquant/nj-quant && .venv/bin/python signal_pipeline/night_backfill.py >> logs/backfill.log 2>&1

# 09:00 — Web 健康检查（周一至周五）
0 9 * * 1-5  systemctl is-active --quiet nj-quant-web || systemctl restart nj-quant-web
```

> `TUSHARE_TOKEN` 也可以写入 `.env` 文件，由 systemd 服务通过 `EnvironmentFile` 加载。

### 步骤 12：验证部署

```bash
# 检查 Web 服务
curl http://localhost:8080/health

# 检查 Nginx
curl http://localhost/

# 检查 cron 是否生效
sudo journalctl -u nj-quant-web -f
```

---

## 每日运维

### 手动触发信号生成（验证用）

```bash
cd /home/njquant/nj-quant
.venv/bin/python signal_pipeline/intraday_signal.py
tail -f logs/intraday.log
```

### 手动触发数据补全

```bash
cd /home/njquant/nj-quant
.venv/bin/python signal_pipeline/night_backfill.py
tail -f logs/backfill.log
```

### 查看 cron 执行记录

```bash
# 通过 Web API
curl http://localhost:8080/api/cron/

# 通过数据库
psql -U quant_user -d quant_db -c "SELECT * FROM cron_log ORDER BY started_at DESC LIMIT 10;"
```

### 服务管理

```bash
# 重启 Web 服务
sudo systemctl restart nj-quant-web

# 查看 Web 服务日志
sudo journalctl -u nj-quant-web -f

# 停止 Web 服务
sudo systemctl stop nj-quant-web
```

### 数据完整性检查

```bash
curl http://localhost:8080/api/cron/status
```

---

## 目录结构

部署后服务器上的目录结构：

```
/home/njquant/nj-quant/
├── .env                          # 环境变量（包含 TUSHARE_TOKEN、DB 密码）
├── .venv/                        # Python 虚拟环境
├── signal_pipeline/              # 信号管线
│   ├── intraday_signal.py       # 14:25 cron 入口
│   ├── night_backfill.py        # 18:00 cron 入口
│   ├── indicator_calculator.py  # 向量化技术指标计算
│   ├── data_merger.py           # 日内数据 + 历史数据合并
│   ├── signal_generator.py      # 双层信号管线
│   └── schema.sql               # 4 张新表
│   └── data_sources/
│       ├── tushare_client.py    # Tushare Pro 客户端
│       └── akshare_client.py    # AKShare 实时快照
├── web/
│   ├── server/                  # FastAPI 后端
│   │   ├── main.py              # 入口 (:8080)
│   │   ├── api/                 # REST API
│   │   │   ├── signals.py       # 信号 CRUD
│   │   │   ├── positions.py     # 持仓 & 资金
│   │   │   ├── data_browser.py  # 数据浏览
│   │   │   └── cron_status.py   # 任务追踪
│   │   └── models/schemas.py    # Pydantic 模型
│   └── frontend/
│       └── dist/                # React 构建产物（Nginx served）
├── cache/
│   └── daily_rotation/
│       ├── daily/               # 每日 Parquet 缓存
│       │   └── {YYYY-MM-DD}.parquet
│       └── trading_dates.parquet
├── logs/
│   ├── intraday.log             # 信号脚本日志
│   └── backfill.log             # 补全脚本日志
├── deploy/                      # 部署配置模板（可删除）
│   ├── nj-quant-web.service
│   ├── nginx.conf
│   ├── crontab.txt
│   └── .env.example
└── scripts/
    ├── start.sh                 # 开发环境一键启动
    ├── server_setup_01_init.sql
    └── server_setup_02_tables.sql
```

---

## 常见问题

### AKShare 获取数据失败

14:30 前 AKShare 调用失败会导致当天无信号。脚本会自动重试 3 次（每次等 60s）。仍失败则写入 `cron_log` 状态 `failed`，Web 界面红色高亮，可次日手动补跑。

### Tushare 请求受限

Tushare Pro 有频率限制，每晚 3-5 次调用足够。`night_backfill.py` 在失败后等待 120s 重试 3 次。积分不足可联系 Tushare 提升。

### 数据库连接失败

检查 `.env` 中 `DB_HOST`、`DB_PORT`、`DB_NAME`、`DB_USER`、`DB_PASSWORD` 是否正确。

```bash
psql -U quant_user -d quant_db -c "SELECT 1;"
```

### Web 服务启动失败

```bash
sudo journalctl -u nj-quant-web -e
```

常见原因：端口 8080 被占用、`.env` 文件不存在、依赖未安装。

### 前端页面空白

检查 Nginx 是否正确指向 `dist/` 目录：

```nginx
root /home/njquant/nj-quant/web/frontend/dist;
```
