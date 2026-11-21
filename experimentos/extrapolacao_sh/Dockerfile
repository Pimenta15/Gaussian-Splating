# Ambiente dos experimentos: 3DGS (gsplat 1.5.3) + Ref-Gaussian (código oficial) sobre CUDA 11.8 / PyTorch 2.0.
# Derivado da imagem usada nos experimentos, sem a infraestrutura do serviço (fila, MLflow, armazenamento,
# segmentação), que nenhum script daqui usa.
#
#   docker build -t vitrine3d-worker:dev .
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # RTX 30xx (8.6) e 40xx (8.9); acrescente a arquitetura da sua GPU se for outra
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    # extensões compiladas na primeira execução (nvdiffrast, gsplat) ficam num volume
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ninja-build build-essential \
        libgl1 libglib2.0-0 libgomp1 libeigen3-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python
# o setup.py do raytracing baixa o Eigen para um caminho que o build do pip não enxerga; usa o do sistema
ENV CPATH=/usr/include/eigen3

RUN pip install --upgrade pip \
    && pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# --- Ref-Gaussian (código oficial, commit fixo) -------------------------------------------
ARG REFGS_COMMIT=af7691aa1097cc6bf89dab6de0fcd358e8830bd8
RUN git clone https://github.com/fudan-zvg/ref-gaussian.git /opt/ref-gaussian \
    && cd /opt/ref-gaussian && git checkout ${REFGS_COMMIT}
RUN cd /opt/ref-gaussian \
    && pip install --no-build-isolation ./submodules/simple-knn ./submodules/cubemapencoder ./submodules/diff-surfel-rasterization
RUN cd /opt/ref-gaussian && pip install --no-build-isolation --no-deps ./submodules/raytracing
RUN pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.1"
RUN pip install plyfile imageio==2.35.1 kornia==0.7.3 mediapy open3d==0.18.0 \
        opencv-python-headless==4.10.0.84 "scipy<1.12" "scikit-image<0.23" trimesh tqdm matplotlib ninja

# --- 3DGS (gsplat) --------------------------------------------------------------------------
# 1.5.3 é a última versão com wheel pt20cu118; o bug de reset de opacidade dela é tratado em train_3dgs.py
RUN pip install numpy==1.26.4 jaxtyping rich \
    && pip install gsplat==1.5.3+pt20cu118 --index-url https://docs.gsplat.studio/whl/pt20cu118
RUN pip install lpips==0.1.4
# o leitor de dados do Ref-Gaussian faz Image.fromarray(..., dtype=np.byte); Pillow >= 11 recusa int8
RUN pip install pillow==10.4.0

ENV REFGS_DIR=/opt/ref-gaussian
WORKDIR /exp
