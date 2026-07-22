@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PPG_BP_DATA_ROOT=D:\Datasets\PPG_BP_Robustness"

:menu
cls
echo ================================================================
echo                    PPG-BP 项目入口
echo ================================================================
echo 1. 查看目前做了什么、下载了什么
echo 2. 打开一段真实PPG波形图
echo 3. 重新运行小型训练演示
echo 4. 打开实验结果文件夹
echo 5. 打开GitHub项目
echo 6. 退出
echo ================================================================
choice /c 123456 /n /m "请输入数字："

if errorlevel 6 goto :eof
if errorlevel 5 start "" "https://github.com/jiangyucheng2006-code/PPG_BP_Robustness" & goto menu
if errorlevel 4 start "" "%~dp0outputs" & goto menu
if errorlevel 3 goto train_demo
if errorlevel 2 start "" "%~dp0outputs\p000160_audit\first_segment.png" & goto menu
if errorlevel 1 goto status

:status
cls
python scripts\project_status.py
echo.
pause
goto menu

:train_demo
cls
echo 正在使用少量真实数据运行XResNet50演示训练……
echo 该结果只用于检查程序，不能作为论文结果。
python scripts\train_baseline.py --config configs\pulsedb_xresnet50_smoke.yaml --output outputs\demo_xresnet50
echo.
echo 运行结束。结果保存在 outputs\demo_xresnet50
pause
goto menu

