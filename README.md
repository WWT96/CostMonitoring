# CostMonitoring 运行指南

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
- 生成示例数据：

```bash
python generate_test_data.py
```

- 项目已包含示例数据文件：`test_data.csv`、`test_data.xlsx`（生成后）

## 常见问题
- 提示找不到 `streamlit`：
  - 执行 `pip install streamlit`
- 端口被占用：
  - 更换端口号，例如 `--server.port 8502`
- 权限或路径问题：
  - 确认在项目根目录执行命令

python -m streamlit run app.py