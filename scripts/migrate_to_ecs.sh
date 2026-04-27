#!/bin/bash
# ============================================================
# 数据迁移脚本: 本地 PostgreSQL → ECS PostgreSQL
#
# 用法（在本地 Windows Git Bash 或 WSL 中执行）:
#   1. 先设置 ECS_IP 环境变量: export ECS_IP=你的服务器IP
#   2. bash migrate_to_ecs.sh
#
# 也可以分步手动执行, 见脚本内注释
# ============================================================

set -e

ECS_IP="${ECS_IP:?请设置 ECS_IP 环境变量, 例如: export ECS_IP=47.xx.xx.xx}"
LOCAL_HOST="localhost"
ECS_HOST="$ECS_IP"
DB_USER="quant_user"
DB_NAME="quant_db"
DB_PASS="123456"

# 设置密码环境变量，避免交互输入
export PGPASSWORD="$DB_PASS"

echo "=== 步骤 1/3: 在服务器上创建数据库 ==="
echo "如果数据库已存在会报错，可以忽略"
psql -h "$ECS_HOST" -U postgres -d postgres <<SQL
SELECT 'CREATE DATABASE quant_db OWNER quant_user;'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'quant_db')\gexec
SQL

echo ""
echo "=== 步骤 2/3: 在服务器上创建表结构 ==="
psql -h "$ECS_HOST" -U "$DB_USER" -d "$DB_NAME" -f scripts/server_setup.sql

echo ""
echo "=== 步骤 3/3: 迁移数据 ==="
echo "正在从本地导出并直接导入到 ECS，请耐心等待..."
pg_dump \
    -h "$LOCAL_HOST" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --no-owner \
    --no-privileges \
    --data-only \
    --inserts \
    | psql -h "$ECS_HOST" -U "$DB_USER" -d "$DB_NAME"

echo ""
echo "=== 迁移完成! ==="
echo "验证: psql -h $ECS_HOST -U $DB_USER -d $DB_NAME -c '\dt'"
echo "验证: psql -h $ECS_HOST -U $DB_USER -d $DB_NAME -c 'SELECT count(*) FROM stock_daily;'"

unset PGPASSWORD
