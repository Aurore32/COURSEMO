from beam import Image, endpoint, Volume, env, QueueDepthAutoscaler, experimental
from tqdm import tqdm

MOUNT_PATH = "./llama-3.3-70b-instruct-bnb-4bit"

@endpoint(
    secrets=["HF_TOKEN"],
    name="llama-3.3-70b-instruct-bnb-4bit",
    timeout=-1,
    volumes=[Volume(name="llama-3.3-70b-instruct-bnb-4bit", mount_path=MOUNT_PATH)],
    cpu=4,
    # We can switch to a smaller, more cost-effective GPU for inference rather than fine-tuning
    gpu=['any'],
    gpu_count=1,
    keep_warm_seconds=600,
    image=Image(
        python_version="python3.12",
        python_packages=["requests", 'tqdm'],
        env_vars=["HF_HUB_ENABLE_HF_TRANSFER=1"]),
    # This autoscaler spawns new containers (up to 5) if the queue depth for tasks exceeds 1
    autoscaler=QueueDepthAutoscaler(max_containers=5, tasks_per_container=1),
)

def download_model():
    import requests
    
    for i in range(1,9):
        url = "https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit/resolve/main/model-0000{}-of-00008.safetensors?download=true".format(i)
        response = requests.get(url)
        total = int(response.headers.get('content-length', 0))
        print(total)
        with open("./model-0000{}-of-00008.safetensors".format(i), "wb") as f, tqdm(desc='model-0000{}-of-00008.safetensors',
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=1024):
                print('update')
                size = f.write(data)
                bar.update(size)
