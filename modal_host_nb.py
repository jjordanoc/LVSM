import os
import subprocess
import time

import modal

app = modal.App(
    image=modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "curl", "unzip", "wget", "git-lfs")
    .pip_install(
        "jupyter"
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/hf_cache",
            HF_HUB_ENABLE_HF_TRANSFER="1",
        )
    )
)
volume = modal.Volume.from_name(
    "modal-jupyter", create_if_missing=True
)

JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!

HOURS = 3600


@app.function(max_containers=1, volumes={"/root/jupyter": volume}, timeout=24 * HOURS, gpu="A100")
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 24 * HOURS):
    run_jupyter.remote(timeout=timeout)

