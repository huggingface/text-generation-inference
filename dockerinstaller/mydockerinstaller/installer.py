import subprocess
import os
import sys


def install_docker():
    try:
        # Check if Docker is installed
        subprocess.run(["docker", "--version"], check=True)
    except subprocess.CalledProcessError:
        # Determine the OS and install Docker
        if os.name == "posix":
            plat = sys.platform
            if plat.startswith("linux"):
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "docker.io"], check=True
                )
            elif plat == "darwin":
                subprocess.run(["brew", "install", "--cask", "docker"], check=True)
            else:
                sys.exit("Unsupported OS for Linux-like environments: {}".format(plat))
        elif os.name == "nt":
            subprocess.run(["choco", "install", "docker-desktop"], check=True)
        else:
            sys.exit("Unsupported OS: {}".format(os.name))


def install_nvidia_toolkit():
    if sys.platform.startswith("linux"):
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(
            ["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"], check=True
        )
    else:
        print("NVIDIA Container Toolkit is not supported on this OS.")


def run_docker_container(model, volume, detach=False):
    subprocess.run(
        [
            "docker",
            "run",
            # conditionally add the -d flag to run the container in the background
            # *(["-d"] if detach else []),
            "-it",
            "--gpus",
            "all",
            "--shm-size",
            "1g",
            "-p",
            "8080:80",
            "-v",
            "{}:/data".format(volume),
            "ghcr.io/huggingface/text-generation-inference:latest",
            "--model-id",
            model,
        ],
        check=True,
    )


def run_detached_docker_container(model, volume):
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-it",
            "--gpus",
            "all",
            "--shm-size",
            "1g",
            "-p",
            "8080:80",
            "-v",
            "{}:/data".format(volume),
            "ghcr.io/huggingface/text-generation-inference:latest",
            "--model-id",
            model,
        ],
        check=True,
    )


def main():
    print("Installing Docker...")
    install_docker()
    install_nvidia_toolkit()
    run_docker_container("HuggingFaceH4/zephyr-7b-beta", os.getcwd() + "/data")


from threading import Thread
import threading


# a class that keeps the container running in a separate thread
class DockerThread(threading.Thread):
    def __init__(self, model, volume):
        threading.Thread.__init__(self)
        self.model = model
        self.volume = volume

    def run(self):
        self._stop = threading.Event()
        run_detached_docker_container(self.model, self.volume)

    def stop(self):
        self._stop.set()


# how to use the DockerThread class
# import os

# docker_thread = DockerThread("HuggingFaceH4/zephyr-7b-beta", os.getcwd() + "/data")

# docker_thread.start()

# do some other stuff


# import subprocess
# import os
# import sys


# def check_command_exists(command):
#     """Check if a command exists in the system's PATH."""
#     return subprocess.run(["which", command], stdout=subprocess.PIPE).returncode == 0


# def install_rust():
#     """Install Rust if it's not already installed."""
#     if not check_command_exists("cargo"):
#         subprocess.run(
#             ["curl", "--proto", "=https", "--tlsv1.2", "-sSf", "https://sh.rustup.rs"],
#             stdout=subprocess.PIPE,
#         )
#         subprocess.run(["sh", "rustup-init", "-y"], check=True)
#     else:
#         print("Rust is already installed.")


# def install_docker():
#     """Install Docker if it's not already installed."""
#     if check_command_exists("docker"):
#         print("Docker is already installed.")
#         return True
#     try:
#         if os.name == "posix":
#             plat = sys.platform
#             if plat.startswith("linux"):
#                 subprocess.run(["sudo", "apt-get", "update"], check=True)
#                 subprocess.run(
#                     ["sudo", "apt-get", "install", "-y", "docker.io"], check=True
#                 )
#             elif plat == "darwin":
#                 subprocess.run(["brew", "install", "--cask", "docker"], check=True)
#             else:
#                 sys.exit(f"Unsupported OS for Linux-like environments: {plat}")
#         elif os.name == "nt":
#             subprocess.run(["choco", "install", "docker-desktop"], check=True)
#         else:
#             sys.exit(f"Unsupported OS: {os.name}")
#         return True
#     except subprocess.CalledProcessError:
#         return False


# def install_nvidia_toolkit():
#     """Install NVIDIA toolkit if on Linux."""
#     if sys.platform.startswith("linux"):
#         subprocess.run(["sudo", "apt-get", "update"], check=True)
#         subprocess.run(
#             ["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"], check=True
#         )
#     else:
#         print("NVIDIA Container Toolkit is not supported on this OS.")


# def clone_and_install(repo_url, directory):
#     """Clone a repository and run 'make install'."""
#     os.chdir(directory)
#     subprocess.run(["git", "clone", repo_url], check=True)
#     os.chdir(repo_url.split("/")[-1])
#     subprocess.run(["make", "install"], check=True)


# def run_docker_container(model, volume):
#     """Run a Docker container with specified parameters."""
#     subprocess.run(
#         [
#             "docker",
#             "run",
#             "-it",
#             "--gpus",
#             "all",
#             "--shm-size",
#             "1g",
#             "-p",
#             "8080:80",
#             "-v",
#             f"{volume}:/data",
#             "ghcr.io/huggingface/text-generation-inference:latest",
#             "--model-id",
#             model,
#         ],
#         check=True,
#     )


# def main():
#     repo_url = "https://github.com/example/repo.git"
#     directory = "/path/to/clone"
#     use_docker = input("Do you want to use Docker? (yes/no): ").strip().lower() == "yes"

#     install_rust()
#     if use_docker:
#         print("Installing Docker...")
#         if install_docker():
#             install_nvidia_toolkit()
#             run_docker_container("HuggingFaceH4/zephyr-7b-beta", os.getcwd() + "/data")
#         else:
#             print("Failed to install Docker. Attempting to build from source.")
#             clone_and_install(repo_url, directory)
#     else:
#         clone_and_install(repo_url, directory)


# if __name__ == "__main__":
#     main()
