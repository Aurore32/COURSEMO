from beam import Pod, Image

pod = Pod(
    name="GRPO",
    image=Image(python_version="python3.12").add_python_packages(["transformers", "torch", "unsloth", "trl", "wandb"]),
    gpu='H100',
    ports=[8000],
    cpu=4,
    memory=16384,
    entrypoint=["python3", "-m", "http.server", "8000"],
)
instance = pod.create()

print("✨ Container hosted at:", instance.url)