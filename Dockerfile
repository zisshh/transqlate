##############################
#  Dockerfile for Transqlate #
##############################

FROM python:3.11.7-slim

# ─── Build-time switches ──────────────────────────────────────────────
# 0 = CPU image • 1 = GPU image
ARG INSTALL_GPU_DEPS=0        
ARG TORCH_VERSION=2.7.0
ARG CUDA_TAG=cu126           
# ───────────────────────────────────────────────────────────────────────

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1

# ─── OS-level deps ─────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git gcc curl gnupg \
        libpq-dev libsqlite3-dev \
        libmariadb-dev libmariadb-dev-compat \
        unixodbc-dev && \
    curl -sSL https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
        -o /tmp/msprod.deb && \
    dpkg -i /tmp/msprod.deb && rm /tmp/msprod.deb && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ─── Copy metadata early (caches better) ───────────────────────────────
COPY pyproject.toml requirements.txt ./

# ─── 1️⃣  Torch first (CUDA or CPU) — ONE download only ────────────────
RUN set -e; \
    if [ "$INSTALL_GPU_DEPS" = "1" ]; then \
        echo "🔧  Installing CUDA Torch (${CUDA_TAG}) …"; \
        pip install torch==${TORCH_VERSION}+${CUDA_TAG} \
            --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} && \
        pip install bitsandbytes==0.46.0; \
        python -c "import torch, textwrap; \
print(textwrap.dedent(f'✓ Torch {torch.__version__} built with CUDA {torch.version.cuda}  (GPU will be visible at runtime)'))"; \
    else \
        echo '🔧  Installing CPU Torch …'; \
        pip install torch==${TORCH_VERSION}; \
    fi

# ─── 2️⃣  Rest of deps once, skipping Torch ────────────────────────────
RUN pip install --requirement requirements.txt --no-deps

# ─── 3️⃣  Package source & install (skip deps again) ───────────────────
COPY src/transqlate ./transqlate
RUN pip install . --no-deps

# ─── Entrypoint ────────────────────────────────────────────────────────
ENTRYPOINT ["transqlate"]
CMD ["--interactive"]