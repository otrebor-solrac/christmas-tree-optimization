FROM rust:latest

ENV DEBIAN_FRONTEND=noninteractive

# 1. Dependencias del sistema:
#    - Bibliotecas gráficas para Macroquad (miniquad): X11, OpenGL, ALSA
#    - Python 3 y utilidades
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libx11-dev \
    libxi-dev \
    libxcursor-dev \
    libxrandr-dev \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libgl1-mesa-dev \
    libasound2-dev \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

# 2. Configurar entorno virtual de Python
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# 3. Instalar librerías de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar scripts y código fuente
COPY . .

# 5. Precompilar Rust e instalar binarios en /usr/local/bin
RUN cd rust_optimizer && cargo build --release \
 && cp target/release/christmas_tree_optimizer /usr/local/bin/ \
 && cp target/release/visualizer /usr/local/bin/

CMD ["/bin/bash"]
