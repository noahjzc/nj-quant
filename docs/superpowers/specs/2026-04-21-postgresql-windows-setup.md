# PostgreSQL Windows 部署指南

**日期**: 2026-04-21
**适用版本**: PostgreSQL 16.x / 17.x

---

## 1. 下载 PostgreSQL

1. 访问 PostgreSQL 官方下载页：
   https://www.postgresql.org/download/windows/

2. 选择 **Windows x86-64** 版本

3. 推荐下载 **Interactive Installer** 版本（EnterpriseDB 提供）

---

## 2. 安装步骤

### 2.1 运行安装程序

双击 `.exe` 文件启动安装向导。

### 2.2 安装组件选择

保持默认即可，包含：
- [x] PostgreSQL Server
- [x] pgAdmin 4（Web 管理工具）
- [x] Command Line Tools

### 2.3 设置目录

```
安装目录: D:\pgsql
数据目录: D:\pgsql\data
```

### 2.4 设置超级用户密码

```
用户名: postgres
密码:  [请设置一个强密码，记录下来]
```

**重要：请记住此密码，后续配置需要使用。**

### 2.5 端口配置

默认端口：`5432`

保持默认即可。

### 2.6 完成安装

点击 Next → Install，等待安装完成。

---

## 3. 配置环境变量

为了方便在命令行使用 PostgreSQL 工具，添加环境变量：

### 3.1 打开系统环境变量设置

1. 右键 **此电脑** → **属性**
2. 点击 **高级系统设置**
3. 点击 **环境变量**

### 3.2 添加 Path 变量

在 **系统变量** 中找到 **Path**，双击编辑：

添加以下路径：
```
D:\pgsql\bin
```

---

## 4. 验证安装

打开 PowerShell 或 CMD，运行：

```powershell
psql --version
```

应显示：
```
psql (PostgreSQL) 16.x
```

---

## 5. 创建数据库和用户

### 5.1 连接 PostgreSQL

```powershell
psql -U postgres -h localhost
```

输入密码后进入 psql 命令行。

### 5.2 创建数据库

```sql
-- 创建数据库
CREATE DATABASE quant_db;
```

### 5.3 创建专用用户

```sql
-- 创建用户（设置密码）
CREATE USER quant_user WITH PASSWORD 'your_password_here';

-- 授权
GRANT ALL PRIVILEGES ON DATABASE quant_db TO quant_user;

-- 授权 schema 权限
GRANT ALL PRIVILEGES ON SCHEMA public TO quant_user;
```

### 5.4 退出

```sql
\q
```

---

## 6. 配置 pgAdmin 4（可选）

pgAdmin 4 是 PostgreSQL 的 Web 管理界面，安装时已包含。

### 6.1 启动 pgAdmin

开始菜单中找到 **pgAdmin 4**

### 6.2 添加服务器

1. 点击 **Add New Server**
2. **General** 标签：
   - Name: `Local PostgreSQL`
3. **Connection** 标签：
   - Host: `localhost`
   - Port: `5432`
   - Database: `quant_db`
   - Username: `quant_user`
   - Password: `[quant_user 的密码]`

### 6.3 验证连接

成功连接后可以在 pgAdmin 中查看数据库结构。

---

## 7. 数据库配置检查清单

| 检查项 | 预期值 |
|--------|--------|
| psql 命令可用 | `psql --version` 显示版本 |
| 数据库 `quant_db` 存在 | pgAdmin 中可见 |
| 用户 `quant_user` 存在 | pgAdmin 中可见 |
| 端口 | 5432 |
| 编码 | UTF-8 |

---

## 8. 防火墙配置（可选）

如果需要远程访问 PostgreSQL，需要配置防火墙：

```powershell
# 允许 PostgreSQL 端口入站
netsh advfirewall firewall add rule name="PostgreSQL" dir=in action=allow protocol=tcp localport=5432
```

---

## 9. 常用命令参考

### 连接数据库
```powershell
psql -U quant_user -d quant_db -h localhost
```

### 列出所有数据库
```sql
\l
```

### 列出所有用户
```sql
\du
```

### 切换数据库
```sql
\c quant_db
```

### 列出所有表
```sql
\dt
```

### 退出
```sql
\q
```

---

## 10. 后续步骤

PostgreSQL 部署完成后，可以开始数据架构的实施：

1. 执行建表脚本
2. 配置 akshare 数据同步
3. 改造 DataProvider

详见：`docs/superpowers/specs/2026-04-21-data-architecture-design.md`
