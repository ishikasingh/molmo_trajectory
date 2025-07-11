#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_URL = "241533154612.dkr.ecr.us-east-1.amazonaws.com"
REPO_NAME = "dexmimicgen"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


def run_command_with_output(command) -> tuple[str, bool]:
    command_text = " ".join(command)
    print(f"-> {GREEN}{command_text}{RESET}")

    # Running the command
    success = True
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        success = False
        output = e.output

    print(f"{BLUE}{output}{RESET}")

    return output, success


def exec_command(command) -> bool:
    command_text = " ".join(command)
    print(f"-> {GREEN}{command_text}{RESET}")

    # Running the command
    success = True
    try:
        output = subprocess.run(command)
    except subprocess.CalledProcessError:
        success = False
        output = None
    except FileNotFoundError:
        success = False
        output = None

    if success and output is not None and output.returncode != 0:
        success = False

    return success


def get_root_folder() -> Path:
    """
    Retrieve the root directory of the codebase using the existing `run_command_with_output` function.
    We assume that we know the relative path between the root and this script, i.e. it's in the docker folder
    """
    # Get the current file's path
    current_file = Path(__file__).resolve()
    git_root = current_file.parent.parent
    return git_root


def full_image_name(image_tag: str) -> str:
    return f"{REPO_URL}/{REPO_NAME}:{image_tag}"


def check_image_exists(repo_name: str, tag: str, repo_url: str | None) -> bool:
    """
    Check if an image exists in the local Docker repository.
    The full image name is expected to be in the format of <repo_url>/<repo_name>:<tag>.
    If repo_url is not provided, then we just check <repo_name>:<tag> in the local Docker repository.
    """
    result = run_command_with_output(["docker", "image", "ls", "--format", "json"])[0].strip()
    
    # The json format example is: 
    # {"Containers":"N/A","CreatedAt":"2024-09-17 22:31:31 +0000 UTC",
    # "CreatedSince":"12 days ago","Digest":"","ID":"2283e8fb53b1",
    # "Repository":"673492702245.dkr.ecr.us-east-2.amazonaws.com/gmp_dev",
    # "SharedSize":"N/A","Size":"54GB","Tag":"\u003cnone\u003e","UniqueSize":"N/A","VirtualSize":"54.03GB"}

    # We need to parse the json and check if the Repository field matches the repo_name and Tag field matches the tag.

    for line in result.split("\n"):
        image_info = json.loads(line)
        if repo_url is None:
            full_repo_name = image_info["Repository"]
        else:
            full_repo_name = f"{repo_url}/{repo_name}"
        if image_info["Repository"] == full_repo_name and image_info["Tag"] == tag:
            return True
    return False


def push_built_image_to_ecr(image_tag: str):
    # Ensure the image exists locally
    if not check_image_exists(REPO_NAME, image_tag, REPO_URL):
        print(f"{RED}Image with repo {REPO_URL}/{REPO_NAME} and tag {image_tag} does not exist. Exiting.{RESET}")
        sys.exit(1)

    # Push the image to the Docker repo
    run_command_with_output(["docker", "push", full_image_name(image_tag)])


def check_if_container_running(container_name):
    # check if container exists and is running:
    running_container_id = run_command_with_output(
        ["docker", "ps", "-q", "-f", f"name={container_name}", "-f", "status=running"]
    )[0].strip()
    return running_container_id != ""


def remove_container_if_not_running(container_name) -> bool:
    running_container_id = run_command_with_output(
        ["docker", "ps", "-q", "-f", f"name={container_name}", "-f", "status=running"]
    )[0].strip()
    if running_container_id != "":
        return False

    container_id = run_command_with_output(["docker", "ps", "-a", "-q", "-f", f"name={container_name}"])[0].strip()
    if container_id != "":
        run_command_with_output(["docker", "rm", container_id])
        return True
    return False


def create_dirs() -> None:
    home_full_path = os.path.expanduser("~")
    dirs_to_make = [
        f"{home_full_path}/project_files/{REPO_NAME}_files/data",
    ]
    for dirname in dirs_to_make:
        os.makedirs(dirname, exist_ok=True)


def get_path_mounts() -> list[tuple[str, str]]:
    home_full_path = os.path.expanduser("~")
    return [
        (get_root_folder(), f"/workspace"),
        (f"{home_full_path}/.cache", "/root/.cache"),
        (f"{home_full_path}/.cache/huggingface", "/root/.cache/huggingface"),
        (f"{home_full_path}/.netrc", "/root/.netrc"), # wandb
        (f"{home_full_path}/project_files/{REPO_NAME}_files/", f"/root/project_files/{REPO_NAME}_files/"),
        (f"{home_full_path}/temp_scripts", "/root/temp_scripts"),
        (f"{home_full_path}/raw_datasets", "/root/raw_datasets"),
        ("/nfs", "/root/nfs"),
    ]


def make_mount_args() -> Iterable[str]:
    for src, target in get_path_mounts():
        if src is None:
            continue
        if not os.path.exists(src):
            print(f"{RED}Directory {src} does not exist. Skipping mount.{RESET}")
            continue

        arg = f"--mount=type=bind,source={src},target={target}"
        yield arg


def system_wide_opts_for_docker_run(shm_size: int, use_gpus: bool) -> Iterable[str]:
    options = []
    if use_gpus:
        options.append('--gpus=all,"capabilities=compute,utility,graphics,display"')
    options = options + [
        "--net=host",  # Required for passing through host ports.
        "--shm-size",
        f"{shm_size}GB",  # This is needed for torch multi-threaded dataloader.
        "-v",
        "/tmp/.X11-unix:/tmp/.X11-unix",
        "-v",
        f"{os.environ.get('HOME')}/.Xauthority:/root/.Xauthority",
        "-v",
        "/usr/share/vulkan:/usr/share/vulkan:ro",
        "-v",
        "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d:ro",
        "-e",
        f"DISPLAY={os.environ.get('DISPLAY', '')}",
        "-e",
        f"ENTRYPOINT=\"{os.environ.get('ENTRYPOINT', '')}\"",
        "-e",
        f"XDG_RUNTIME_DIR={os.environ.get('XDG_RUNTIME_DIR', '')}",
    ]

    # Pass skypilot environment variables through to the container.
    for key, value in os.environ.items():
        if key.startswith("SKYPILOT_"):
            options.extend(["-e", f"{key}={value}"])

    return options


def main():
    parser = argparse.ArgumentParser(description="Run the docker container")
    parser.add_argument("--force-build", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-local-image", action=argparse.BooleanOptionalAction, help="Use the local image", default=False)
    parser.add_argument("--push-built-image", action=argparse.BooleanOptionalAction, help="Push the image to the Docker repo")
    parser.add_argument("--image-tag", type=str, default="groot", help="Tag of the image to use")
    parser.add_argument("--shm-size", type=int, default=64, help="Size of the shared memory in GB")
    parser.add_argument("--cmd", type=str, default=None, help="Command of the container")
    parser.add_argument("--use-gpus", dest="use_gpus", action="store_true", default=None, help="Use GPUs")
    parser.add_argument("--no-gpus", dest="use_gpus", action="store_false", default=None, help="Do not use GPUs")
    parser.add_argument("--detach", action=argparse.BooleanOptionalAction, help="Detach the container", default=False)
    args = parser.parse_args()

    password = run_command_with_output(["aws", "ecr", "get-login-password", "--region", "us-east-1"])[0]
    subprocess.run(["docker", "login", "--username", "AWS", "--password-stdin", REPO_URL], input=password.encode("utf-8"))

    if args.force_build:
        sucesss = exec_command(
            [
                "docker",
                "build",
                "--progress",
                "plain",
                "--tag",
                full_image_name(args.image_tag),
                "--file",
                "Dockerfile",
                ".",
            ]
        )

    if args.push_built_image:
        push_built_image_to_ecr(args.image_tag)

    if not args.use_local_image:
        # Pull the docker image from the Docker repo
        _ = exec_command(["docker", "pull", full_image_name(args.image_tag)])

    if args.use_gpus is None:
        use_gpus = exec_command(["nvidia-smi"])
    else:
        use_gpus = args.use_gpus

    create_dirs()

    container_name = f"{os.getenv('USER')}-{REPO_NAME}-{args.image_tag}"
    if check_if_container_running(container_name):
        # If container exists, use docker exec
        exec_command(
            [
                "docker",
                "exec",
                *(["-it"] if sys.stdin.isatty() else []),
                container_name,
                "bash",
            ]
        )
    else:
        remove_container_if_not_running(container_name)
        # If container does not exist, use docker run
        exec_command(
            [
                "docker",
                "run",
                *(["-d"] if args.detach else []),
                "--name",
                container_name,
                *(["-it"] if sys.stdin.isatty() and args.cmd is None else []),
                *list(make_mount_args()),
                *list(system_wide_opts_for_docker_run(shm_size=args.shm_size, use_gpus=use_gpus)),
                full_image_name(args.image_tag),
                *([args.cmd] if args.cmd is not None else [])
            ]
        )

if __name__ == "__main__":
    main()
