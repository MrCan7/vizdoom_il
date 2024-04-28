# Use Debian Bookworm as the base image
FROM debian:bookworm

# Set non-interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive

# Update the system and install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    libbz2-dev \
    libgtk2.0-dev \
    cmake \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    timidity \
    libwildmidi-dev \
    libboost-all-dev \
    x11-apps \
    libx11-dev \
    software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_24.3.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Set up the environment path
ENV PATH /opt/conda/bin:$PATH

# Create a Conda environment with Python 3.11
RUN conda create -n myenv python=3.11 -y && \
    conda clean -afy

# Configure shell to activate conda environment by default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Clone ViZDoom and checkout the specific version
RUN git clone https://github.com/mwydmuch/ViZDoom.git && \
    cd ViZDoom && \
    git checkout tags/1.2.3 -b version-1.2.3 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=OFF .. && \
    make

# Install ViZDoom Python bindings
RUN cd ViZDoom && \
    pip install .

# Set the working directory
WORKDIR /workspace

# Copy the project files into the container
COPY . /workspace

# Optional: Expose ports (e.g., for a web server)
EXPOSE 5029

# Default command to keep the container running
CMD ["sleep", "infinity"]
