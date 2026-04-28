-- ============================================================
-- ECS 初始化步骤 1: 创建用户和数据库（用 postgres 超级用户执行）
-- 在服务器上执行:
--   sudo -u postgres psql -f scripts/server_setup_01_init.sql
-- ============================================================

-- 创建用户
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'quant_user') THEN
        CREATE ROLE quant_user WITH LOGIN PASSWORD 'ZywvF1I4P5YE';
    END IF;
END
$$;

-- 创建数据库（CREATE DATABASE 不能放在条件判断里，直接执行）
-- 如果已存在会报错，可以忽略
CREATE DATABASE quant_db OWNER quant_user;

-- 授权
GRANT ALL PRIVILEGES ON DATABASE quant_db TO quant_user;

