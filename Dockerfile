# 使用官方轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统基础依赖（防止部分 PDF/Word 解析库编译报错）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 库
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制当前目录下的所有源码到容器中
COPY . .

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 配置健康检查
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动命令：运行 Streamlit 应用
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]