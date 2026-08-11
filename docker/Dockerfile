# Humanoid Project — Isaac Sim 5.1 + Isaac Lab 2.3.2 (pip-based, no NGC monolith)
# Build (deliberately, ~15GB):  docker build -f docker/Dockerfile -t humanoid:latest .
# Run:  docker run --gpus all --rm -it -e OMNI_KIT_ACCEPT_EULA=YES humanoid:latest

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PYTHONUNBUFFERED=1

# Isaac Sim runtime libs (headless rendering via EGL/Vulkan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pinned sim stack (single fat layer; keep pins in sync with docs/SETUP.md)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir "isaaclab[isaacsim,all]==2.3.2" \
        --extra-index-url https://pypi.nvidia.com

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY humanoid ./humanoid
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir -e .

CMD ["bash"]
