# Start with Debian Bookworm as the base image
FROM debian:bookworm

# Avoid prompts from apt and other commands
ENV DEBIAN_FRONTEND=noninteractive

# Install basic utilities and dependencies for Miniconda and ViZDoom
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    mercurial \
    subversion \
    procps \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    cmake \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    timidity \
    libwildmidi-dev \
    unzip \
    libboost-all-dev \
    software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add Conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Create a Conda environment with Python 3.11
RUN conda create -n myenv python=3.11 -y && \
    conda clean -afy

# Make RUN commands use the new environment by default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Clone and build ViZDoom
RUN git clone https://github.com/mwydmuch/ViZDoom.git && \
    mkdir -p ViZDoom/build && \
    cd ViZDoom/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=OFF .. && \
    make

# Install ViZDoom Python bindings
RUN cd ViZDoom && \
    pip install .

# Set the working directory in the container
WORKDIR /app

# Copy your application into the container
COPY . /app

# Expose any necessary ports (Optional)
EXPOSE 8000

# Define default command to run your application using Conda environment
CMD ["conda", "run", "-n", "myenv", "python", "app.py"]
