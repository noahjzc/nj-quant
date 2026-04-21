@echo off
REM Windows 定时任务配置脚本
REM 用管理员权限运行

echo 正在创建定时任务...

REM 14:30 盘中更新任务
schtasks /create /tn "Quant-Intraday-Update" ^
    /tr "python D:\workspace\code\mine\quant\nj-quant\back_testing\data\sync\daily_update.py --mode intraday --portfolio sh600519,sh600036" ^
    /sc daily /st 14:30 ^
    /f

REM 15:30 收盘后更新任务
schtasks /create /tn "Quant-Close-Update" ^
    /tr "python D:\workspace\code\mine\quant\nj-quant\back_testing\data\sync\daily_update.py --mode close" ^
    /sc daily /st 15:30 ^
    /f

echo 定时任务创建完成
schtasks /query /tn "Quant-Intraday-Update"
schtasks /query /tn "Quant-Close-Update"

pause
