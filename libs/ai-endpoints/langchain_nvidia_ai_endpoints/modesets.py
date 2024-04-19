catalog_base = "https://integrate.api.nvidia.com/v1"
openai_base = "https://api.openai.com/v1"  ## OpenAI Main URL
nvcf_base = "https://api.nvcf.nvidia.com/v2/nvcf"  ## NVCF Main URL
nvcf_infer = "{base_url}/pexec/functions/{model_id}"  ## Inference endpoints
nvcf_status = "{base_url}/pexec/status/{request_id}"  ## 202 wait handle
nvcf_models = "{base_url}/functions"  ## Model listing
open_models = "{base_url}/models"


MODESET = {
    "nvidia": {
        "base_url": nvcf_base,
        "endpoints": {
            "infer": nvcf_infer,  ## Per-model inference
            "status": nvcf_status,  ## 202 wait handle
            "models": nvcf_models,  ## Model listing
        },
        "api_key": "NVIDIA_API_KEY",
        "api_start": "nvapi-",
    },
    "catalog": {
        "base_url": catalog_base,
        "endpoints": {
            "models": nvcf_models,  ## Model listing
        },
        "api_key": "NVIDIA_API_KEY",
        "api_start": "nvapi-",
    },
    "nim": {
        "endpoints": {
            "models": open_models,  ## Model listing
        },
    },
    "open": {
        "endpoints": {
            "models": open_models,  ## Model listing
        },
    },
    "openai": {
        "base_url": openai_base,
        "endpoints": {
            "models": open_models,  ## Model listing
        },
    },
}
