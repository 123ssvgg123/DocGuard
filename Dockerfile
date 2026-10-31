# ============================================================
#  食安通 · 食品安全知识库系统 —— 镜像构建文件
# ============================================================
#  与 FoodAgent 的做法保持一致：需求 4.6 要求「总部云 + 门店平板」的部署形态，
#  这里给出云侧（服务 + Milvus + Redis）的容器化产物。
#
#  构建：
#      docker build -t foodsafety-system:latest .
#  运行（推荐直接用 docs/deploy/docker-compose.yml，会一并起 Milvus 与 Redis）：
#      docker run --rm -p 8080:8080 \
#        -v %cd%/models:/app/models:ro \
#        -v %cd%/data:/app/data \
#        --env-file .env foodsafety-system:latest
#
#  ⚠️ 三个必须知道的点：
#      1) models/（bge-m3 2.14GB + bge-reranker-large 2.09GB）**不打进镜像**，
#         用 volume 挂载。否则镜像体积会超过 5GB，且改模型要重建镜像。
#      2) torch 的 CUDA / CPU 构建取决于宿主机，官方要求走独立 index-url 安装，
#         故用 ARG TORCH_INDEX_URL 在构建时传入（默认 CPU 版，镜像更小、通用性更好）。
#      3) 容器内 device 建议显式设为 cpu：容器默认拿不到 GPU（除非装了 nvidia 运行时），
#         而 config.ini 里 device=cpu 正好一致，无需改动。
# ============================================================
FROM python:3.10-slim

# 中文分词与 PDF 解析在 slim 镜像上可能需要的基础库
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# ---------- 依赖层（先装依赖，利用镜像层缓存）----------
COPY requirements.txt ./
# torch 单独装：按构建参数选择 CUDA / CPU 源
ARG TORCH_VERSION=2.10.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip \
    && pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && pip install -r requirements.txt

# ---------- 代码层 ----------
COPY base/ ./base/
COPY core/ ./core/
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY app.py qa_system.py build_kb.py build_dataset.py train_intent.py run_cli.py config.ini ./
COPY docs/ ./docs/

# models/ 与 data/ 通过 volume 挂载，这里只建空目录避免启动时报路径不存在
RUN mkdir -p /app/models /app/data /app/logs

EXPOSE 8080

# 健康检查：/health 是 FoodAgent 启动期探活用的同一入口
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

# 说明：app.py 内部用 uvicorn.run(reload=False) 起服务，
#       模型加载在模块导入期完成（进程内单例），start-period 因此给到 90 秒。
CMD ["python", "app.py"]
