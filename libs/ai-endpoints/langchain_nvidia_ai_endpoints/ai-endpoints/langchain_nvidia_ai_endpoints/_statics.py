import os
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, root_validator, Field


class Metadata(BaseModel):
    infer_args: dict = Field({})
    client_args: dict = Field({})
        

class Model(BaseModel):
    id: str
    model_type: Optional[str] = None
    metadata: Optional[Metadata] = None
    # path: str

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        all_required_field_names = {field.alias for field in cls.__fields__.values() if field.alias != 'metadata'}
        client_args = ["client", "base_url", "infer_path", "api_key_var", "api_type", "mode"]
        client_kw: Dict[str, Any] = {}
        infer_kw: Dict[str, Any] = {}
        for field_name in list(values):
            if field_name in client_args:
                client_kw[field_name] = values.pop(field_name)
            elif field_name not in all_required_field_names:
                infer_kw[field_name] = values.pop(field_name)
        values['metadata'] = Metadata(client_args=client_kw, infer_args=infer_kw)
        return values


NVCF_PG_SPECS = {
    "playground_smaug_72b": {"model_type": "chat", "api_type": "aifm"},
    "playground_kosmos_2": {"model_type": "vlm", "api_type": "aifm"},
    "playground_llama2_70b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-llama2-70b",
    },
    "playground_nvolveqa_40k": {"model_type": "embedding", "api_type": "aifm"},
    "playground_nemotron_qa_8b": {"model_type": "qa", "api_type": "aifm"},
    "playground_gemma_7b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-gemma-7b",
    },
    "playground_mistral_7b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-mistral-7b-instruct-v2",
    },
    "playground_mamba_chat": {"model_type": "chat", "api_type": "aifm"},
    "playground_phi2": {"model_type": "chat", "api_type": "aifm"},
    "playground_sdxl": {"model_type": "genai", "api_type": "aifm"},
    "playground_nv_llama2_rlhf_70b": {"model_type": "chat", "api_type": "aifm"},
    "playground_neva_22b": {
        "model_type": "vlm",
        "api_type": "aifm",
        "alternative": "ai-neva-22b",
    },
    "playground_yi_34b": {"model_type": "chat", "api_type": "aifm"},
    "playground_nemotron_steerlm_8b": {"model_type": "chat", "api_type": "aifm"},
    "playground_cuopt": {"model_type": "cuopt", "api_type": "aifm"},
    "playground_llama_guard": {"model_type": "classifier", "api_type": "aifm"},
    "playground_starcoder2_15b": {"model_type": "completion", "api_type": "aifm"},
    "playground_deplot": {
        "model_type": "vlm",
        "api_type": "aifm",
        "alternative": "ai-google-deplot",
    },
    "playground_llama2_code_70b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-codellama-70b",
    },
    "playground_gemma_2b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-gemma-2b",
    },
    "playground_seamless": {"model_type": "translation", "api_type": "aifm"},
    "playground_mixtral_8x7b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-mixtral-8x7b-instruct",
    },
    "playground_fuyu_8b": {
        "model_type": "vlm",
        "api_type": "aifm",
        "alternative": "ai-fuyu-8b",
    },
    "playground_llama2_code_34b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-codellama-70b",
    },
    "playground_llama2_code_13b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-codellama-70b",
    },
    "playground_steerlm_llama_70b": {"model_type": "chat", "api_type": "aifm"},
    "playground_clip": {"model_type": "similarity", "api_type": "aifm"},
    "playground_llama2_13b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-llama2-70b",
    },
}

NVCF_AI_SPECS = {
    "ai-codellama-70b": {"model_type": "chat", "model_name": "meta/codellama-70b"},
    "ai-embed-qa-4": {"model_type": "embedding", "model_name": "NV-Embed-QA"},
    "ai-fuyu-8b": {"model_type": "vlm"},
    "ai-gemma-7b": {"model_type": "chat", "model_name": "google/gemma-7b"},
    "ai-google-deplot": {"model_type": "vlm"},
    "ai-llama2-70b": {"model_type": "chat", "model_name": "meta/llama2-70b"},
    "ai-microsoft-kosmos-2": {"model_type": "vlm"},
    "ai-mistral-7b-instruct-v2": {
        "model_type": "chat",
        "model_name": "mistralai/mistral-7b-instruct-v0.2",
    },
    "ai-mixtral-8x7b-instruct": {
        "model_type": "chat",
        "model_name": "mistralai/mixtral-8x7b-instruct-v0.1",
    },
    "ai-neva-22b": {"model_type": "vlm"},
    "ai-rerank-qa-mistral-4b": {
        "model_type": "ranking",
        "model_name": "nv-rerank-qa-mistral-4b:1",  # nvidia/rerank-qa-mistral-4b
    },
    # 'ai-sdxl-turbo': {'model_type': 'genai'},
    # 'ai-stable-diffusion-xl-base': {'model_type': 'iamge_out'},
    "ai-codegemma-7b": {"model_type": "chat", "model_name": "google/codegemma-7b"},
    "ai-recurrentgemma-2b": {
        "model_type": "chat",
        "model_name": "google/recurrentgemma-2b",
    },
    "ai-gemma-2b": {"model_type": "chat", "model_name": "google/gemma-2b"},
    "ai-mistral-large": {
        "model_type": "chat",
        "model_name": "mistralai/mistral-large",
    },
    "ai-mixtral-8x22b": {
        "model_type": "completion",
        "model_name": "mistralai/mixtral-8x22b-v0.1",
    },
    "ai-mixtral-8x22b-instruct": {
        "model_type": "chat",
        "model_name": "mistralai/mixtral-8x22b-instruct-v0.1",
    },
    "ai-llama3-8b": {"model_type": "chat", "model_name": "meta/llama3-8b-instruct"},
    "ai-llama3-70b": {
        "model_type": "chat",
        "model_name": "meta/llama3-70b-instruct",
    },
    "ai-phi-3-mini": {
        "model_type": "chat",
        "model_name": "microsoft/phi-3-mini-128k-instruct",
    },
    "ai-arctic": {"model_type": "chat", "model_name": "snowflake/arctic"},
    "ai-dbrx-instruct": {
        "model_type": "chat",
        "model_name": "databricks/dbrx-instruct",
    },
}

CATALOG_SPECS = {
    'databricks/dbrx-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'google/codegemma-7b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/gemma-2b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/gemma-7b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/recurrentgemma-2b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/codellama-70b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama2-70b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama3-70b-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama3-8b-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'microsoft/phi-3-mini-128k-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'microsoft/phi-3-mini-4k-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'mistralai/mistral-7b-instruct-v0.2': {'model_type': 'chat', 'max_tokens': 2048},
    'mistralai/mistral-large': {'model_type': 'chat', 'max_tokens': 2048}, 
    'mistralai/mixtral-8x22b-instruct-v0.1': {'model_type': 'chat', 'max_tokens': 2048},
    # 'mistralai/mixtral-8x22b-v0.1': {'model_type': 'chat', 'max_tokens': 2048},
    'mistralai/mixtral-8x7b-instruct-v0.1': {'model_type': 'chat', 'max_tokens': 2048},
    'seallms/seallm-7b-v2.5': {'model_type': 'chat', 'max_tokens': 2048},
    'snowflake/arctic': {'model_type': 'chat', 'max_tokens': 2048},
    'nvidia/embed-qa-4': {'model_type': 'embedding', 'infer_path': '{base_url}/retrieval/nvidia', 'model_name': 'NV-Embed-QA'},
    'snowflake/arctic-embed-l': {'model_type': 'embedding', 'infer_path': '{base_url}/retrieval/snowflake/arctic-embed-l'},
    'adept/fuyu-8b': {'model_type': 'vlm'},
    'google/deplot': {'model_type': 'vlm'},
    'microsoft/kosmos-2': {'model_type': 'vlm'},
    'nvidia/neva-22b': {'model_type': 'vlm'},
    "stabilityai/stable-diffusion-xl": {'model_type': 'genai'},
    "stabilityai/sdxl-turbo": {'model_type': 'genai'},
    "stabilityai/stable-video-diffusion": {'model_type': 'genai'},
}

#      "host": "ai.api.nvidia.com",
 #      "paths": {
 #        "/v1/internal/sre/simple_int8/synthetic": "4ffd8c1e-7bbb-4021-8317-a3e92b318a8a",
 #        "/v1/internal/chipnemo/mixtral_8x7b_H100/chat/completions": "a23b029b-1731-437e-a975-37b961259ddb",
 #        "/v1/internal/chipnemo/codellama2-70b/chat/completions": "b65ceaed-6acc-4fb5-bd3a-8bce0458af1a",
 #        "/v1/internal/chipnemo/chipnemo_8x7b/chat/completions": "51d9b019-5480-4748-92b0-c7b049215524",
 #        "/v1/internal/chipnemo/chipllama_70b_steerlm/chat/completions": "f64ebe04-7cc9-4ce2-8f19-55aff3692a1b",
 #        "/v1/internal/chipnemo/chipllama_70b_sft/chat/completions": "5d29944d-c357-43f6-b073-3c6250161dd9",
	# "/v1/internal/chipnemo/llama3-70b-instruct/chat/completions": "96dc8081-1661-4fb5-8b1f-f665026f02f1",
 #        "/v1/retrieval/nvidia/embeddings": "09c64e32-2b65-4892-a285-2f585408d118",
 #        "/v1/retrieval/nvidia/reranking": "0bf77f50-5c35-4488-8e7a-f49bb1974af6",
 #        "/v1/stg/retrieval/snowflake/arctic-embed-l/embeddings": "bbfd083b-e5bf-4004-b3e2-b5f1744544c6",
 #        "/v1/retrieval/snowflake/arctic-embed-l/embeddings": "1528a0ad-205a-46ac-a783-94e2372586a9",
 #        "/v1/genai/stabilityai/stable-diffusion-xl": "c1b63bb0-448b-4e53-b2a7-fb0b3723cbe2",
 #        "/v1/genai/stabilityai/sdxl-turbo": "f886140c-424e-4c82-a841-99e23f9ae35d",
 #        "/v1/genai/stabilityai/stable-video-diffusion": "8cd594f1-6a4d-4f8f-82b4-d1bf89adae98",
 #        "/v1/vlm/adept/fuyu-8b": "e598bfc1-b058-41af-869d-556d3c7e1b48",
 #        "/v1/vlm/google/deplot": "784a8ca4-ea7d-4c93-bb46-ec027c3fae47",
 #        "/v1/vlm/microsoft/kosmos-2": "6018fed7-f227-48dc-99bc-3fd4264d5037",
 #        "/v1/vlm/nvidia/neva-22b": "bc205f8e-1740-40df-8d32-c4321763498a",
 #        "/v1/llm/cohere/command-r/chat": "97c23efa-0203-4f7d-bd30-997782a9e6d5",
 #        "/v1/llm/cohere/command-r-plus/chat": "7e51a775-3ac3-4cbb-8e9c-c3fa2b48b6b5",
 #        "/v1/stg/retrieval/nvidia/embeddings": "bf928238-9bb2-4efd-aec1-8e1463c76e04",
 #        "/v1/stg/retrieval/nvidia/reranking": "0dc1d6c2-a35b-4735-99f5-6ff8143a157e",
 #        "/v1/stg/genai/stabilityai/stable-diffusion-xl": "d61aa612-c74e-4901-8ec5-40595659cd26",
 #        "/v1/stg/genai/stabilityai/sdxl-turbo": "b04ceaeb-5de6-4a00-94f4-29f19c401861",
 #        "/v1/stg/genai/stabilityai/stable-video-diffusion": "6f1e67d2-e9e7-4182-8f24-c5e30daf3002",
 #        "/v1/stg/vlm/adept/fuyu-8b": "9ec43a4e-e27d-4fb2-9a50-1cd0986f1e1e",
 #        "/v1/stg/vlm/google/deplot": "354d02ef-9d8e-4062-9ae3-edbeaad10833",
 #        "/v1/stg/vlm/microsoft/kosmos-2": "261a56bf-ee3b-4f8a-8c7f-f61c78d8a49b",
 #        "/v1/stg/vlm/nvidia/neva-22b": "3a372d66-82ff-4ad0-94da-14744a91d4c4",
 #        "/v1/stg/llm/cohere/command-r/chat": "27956c6e-dbc8-43ad-8868-ed721eff83d8",
 #        "/v1/stg/llm/cohere/command-r-plus/chat": "287144d0-4a5d-4c3a-9a92-5e9169dafd71"
 #      }

OPENAI_SPECS = {
    "babbage-002": {"model_type": "completion"},
    "dall-e-2": {"model_type": "genai"},
    "dall-e-3": {"model_type": "genai"},
    "davinci-002": {"model_type": "completion"},
    "gpt-3.5-turbo-0125": {"model_type": "chat"},
    "gpt-3.5-turbo-0301": {"model_type": "chat"},
    "gpt-3.5-turbo-0613": {"model_type": "chat"},
    "gpt-3.5-turbo-1106": {"model_type": "chat"},
    "gpt-3.5-turbo-16k-0613": {"model_type": "chat"},
    "gpt-3.5-turbo-16k": {"model_type": "chat"},
    "gpt-3.5-turbo-instruct-0914": {"model_type": "completion"},
    "gpt-3.5-turbo-instruct": {"model_type": "completion"},
    "gpt-3.5-turbo": {"model_type": "chat"},
    "gpt-4-0125-preview": {"model_type": "chat"},
    "gpt-4-0613": {"model_type": "chat"},
    "gpt-4-1106-preview": {"model_type": "chat"},
    "gpt-4-turbo-preview": {"model_type": "chat"},
    "gpt-4-vision-preview": {"model_type": "chat"},
    "gpt-4": {"model_type": "chat"},
    "text-embedding-3-large": {"model_type": "embedding"},
    "text-embedding-3-small": {"model_type": "embedding"},
    "text-embedding-ada-002": {"model_type": "embedding"},
    "tts-1-1106": {"model_type": "tts"},
    "tts-1-hd-1106": {"model_type": "tts"},
    "tts-1-hd": {"model_type": "tts"},
    "tts-1": {"model_type": "tts"},
    "whisper-1": {"model_type": "asr"},
}

CLIENT_MAP = {
    "asr": "None",
    "chat": "ChatNVIDIA",
    "classifier": "None",
    "completion": "NVIDIA",
    "cuopt": "None",
    "embedding": "NVIDIAEmbeddings",
    "vlm": "ChatNVIDIA",
    "genai": "ImageGenNVIDIA",
    "qa": "ChatNVIDIA",
    "similarity": "None",
    "translation": "None",
    "tts": "None",
    "ranking": "NVIDIARerank",
}

for model_name, model_spec in CATALOG_SPECS.items():
    if model_spec.get('model_type') == 'vlm':
        model_spec['base_url'] = 'https://ai.api.nvidia.com/v1'
        model_spec['infer_path'] = '{base_url}/vlm/{model_name}'
    if model_spec.get('model_type') == 'genai':
        model_spec['base_url'] = 'https://ai.api.nvidia.com/v1'
        model_spec['infer_path'] = '{base_url}/genai/{model_name}'

tooled_models = ["gpt-4"]

SPEC_LIST = [CATALOG_SPECS, OPENAI_SPECS, NVCF_PG_SPECS, NVCF_AI_SPECS]
MODE_LIST = ["nvidia", "openai", "nvcf", "nvcf"]

for spec, mode in zip(SPEC_LIST, MODE_LIST):
    for model_name, model_spec in spec.items():
        model_spec["mode"] = model_spec.get("mode") or mode
        ## Default max_tokens for models
        if model_spec.get('model_type') in ('chat', 'vlm'):
            if 'max_tokens' not in model_spec:
                model_spec['max_tokens'] = 1024
        ## Default Client enforcement
        model_spec['client'] = model_spec.get("client") or [CLIENT_MAP.get(model_spec.get("model_type"))]
        if not isinstance(model_spec['client'], list):
            model_spec['client'] = [model_spec['client']]
        if model_name in tooled_models:
            model_spec['client'] += [f"Tooled{client}" for client in model_spec['client']]

NVCF_SPECS = {**NVCF_PG_SPECS, **NVCF_AI_SPECS}

OPEN_SPECS = {**CATALOG_SPECS, **OPENAI_SPECS}

MODEL_SPECS = {**OPEN_SPECS, **NVCF_SPECS}