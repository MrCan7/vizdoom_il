# ViZDoom Development Container Setup

This project uses a Docker-based development environment managed through Visual Studio Code's Remote - Containers extension. This setup ensures a consistent development environment for all contributors, avoiding the "works on my machine" problem.

## Prerequisites

Before you begin, make sure you have the following installed:
- **Docker**: Ensure Docker Desktop (Windows/Mac) or Docker Engine (Linux) is installed and running. [Get Docker](https://docs.docker.com/get-docker/)
- **Visual Studio Code**: Download and install VS Code if you haven't already. [Download VS Code](https://code.visualstudio.com/Download)
- **Remote - Containers Extension**: Install the "Remote - Containers" extension in VS Code. You can find it by searching for "Remote - Containers" in the VS Code Extensions View (`Ctrl+Shift+X`).

## Getting Started

Follow these steps to set up and start using the development environment:

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/MrCan7/vizdoom_il.git
cd vizdoom_il
```

### 2. Open in Visual Studio Code

Open the cloned repository in Visual Studio Code. You can do this from the command line by running:

```bash
code .
```

### 3. Reopen in Container

Once VS Code is open, you'll be prompted to reopen the project in a container. You can also manually reopen the project in the container by:
- Opening the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS).
- Typing "Remote-Containers: Reopen in Container" and pressing Enter.

VS Code will build the Docker image based on the provided `Dockerfile` and `devcontainer.json` configurations and then start a container.

### 4. Work Inside the Container

After the container is set up, you can start coding right away. The development environment, including all dependencies and configurations, will be exactly as specified in the Docker setup.

## Using the DevContainer Without Visual Studio Code

If you need to use the Docker environment without VS Code, you can do so by building and running the container directly from the command line:

### Build the Docker Image

```bash
docker build -t vizdoom-dev .
```

### Run the Docker Container

```bash
docker run -it \
  -e DISPLAY=${DISPLAY} \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  -p 5029:5029 \
  --name vizdoom-container \
  --user dev \ 
  vizdoom-dev
```

This command starts the container with the necessary settings for X11 forwarding (for GUI applications), binds the necessary ports, and ensures you are operating within the designed environment.

# Managing ViZDoom Docker Container

This guide covers the basic commands needed to manage and interact with the ViZDoom Docker container. The container is configured to use a non-root user, `dev`, for executing commands.

## Prerequisites

- Docker must be installed on your system.
- Ensure the ViZDoom container (`vizdoom-container`) is built and running.

## Basic Docker Commands

### Accessing the Container

To open a bash shell inside the container as the non-root user `dev`, use the following command:

```bash
docker exec -it --user dev vizdoom-container /bin/bash
```

### Stopping the Container

To stop the running container, use the following command:

```bash
docker stop vizdoom-container
```

### Starting the Container

If your container is stopped and you wish to start it again, use the following command:

```bash
docker start vizdoom-container
```

### Removing the Container

To completely remove the container (note: this does not remove the Docker image), use the following command:

```bash
docker rm vizdoom-container
```

## Features of the Development Environment

- **ViZDoom**: The ViZDoom environment is fully set up and ready to be used or developed upon.
- **Python**: The container includes a Conda environment with Python 3.11, optimized for working with ViZDoom.
- **Tools and Extensions**: Pre-installed VS Code extensions and tools for Python development, CMake project management, Git integration, and Docker management.

## Additional Commands

To build the Docker image manually or to rebuild it, you can use:

```bash
docker build -t vizdoom-dev .
```

To run the container manually (without VS Code), use:

```bash
docker run -d --name my-vizdoom-dev -p 5029:5029 vizdoom-dev
```

## Troubleshooting

If you encounter any issues with the DevContainer setup, try the following:
- **Rebuild Container**: Open the Command Palette and select "Remote-Containers: Rebuild Container".
- **Check Docker**: Ensure Docker is running correctly on your machine. Restart Docker if necessary.
- **Logs**: Check the Docker and VS Code logs for any error messages that might provide more insight.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for any improvements or bug fixes.
