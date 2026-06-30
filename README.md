# CostMonitoring 运行指南

## 本地架构说明
- 系统配置文件：`settings.json`
- 本地数据库：`cost_monitor_data.db`
- 本地日志目录：`logs/`
- 启动后可在“⚙️ 系统设置”中维护数据路径、LLM Key / Base URL，并执行数据库压缩

## 环境准备
- 确保已安装 Python（建议 3.11+）和 pip
- 切换到项目目录：`c:\Users\Francesco\Documents\trae_projects\CostMonitoring`

## 安装依赖
- 在终端或 CMD 执行：

```bash
pip install -r requirements.txt
```

## 启动应用
- 在终端或 CMD 执行：

```bash
python -m streamlit run app.py
```

- 或者（等效）：

```bash
streamlit run app.py
```

- 启动成功后打开浏览器访问：`http://localhost:8501`

## 在 CMD 中运行（逐步）
- 打开 CMD
- 切换到项目目录（注意要输入 cd）：

```cmd
cd c:\Users\Francesco\Documents\trae_projects\CostMonitoring
```

- 安装依赖：

```cmd
pip install -r requirements.txt
```

- 启动应用并预览：

```cmd
python -m streamlit run app.py
```

- 如果不想切换目录，也可以直接指定文件路径运行：

```cmd
streamlit run c:\Users\Francesco\Documents\trae_projects\CostMonitoring\app.py
```

## 常用选项
- 指定端口（默认 8501）：

```bash
python -m streamlit run app.py --server.port 8501
```

- 停止服务：在运行该命令的终端按下 `Ctrl + C`

## 测试数据（可选）
- 项目当前保留示例数据文件：`test_data.csv`

## 常见问题
- 提示找不到 `streamlit`：
  - 执行 `pip install streamlit`
- 端口被占用：
  - 更换端口号，例如 `--server.port 8502`
- 权限或路径问题：
  - 确认在项目根目录执行命令

python -m streamlit run app.py