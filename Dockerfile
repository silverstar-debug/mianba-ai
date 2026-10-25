# 面霸 AI —— 生产镜像
# 用于部署到 Railway / Render / Fly.io / 任意云服务器

FROM python:3.12-slim

WORKDIR /app

# 依赖先装，利用构建缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码（.env 不打进镜像，密钥通过平台环境变量注入）
COPY main.py index.html ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MAX_ROUNDS=4 \
    DAILY_LIMIT_PER_IP=80

EXPOSE 8000

# 云平台通常通过 $PORT 注入端口
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
