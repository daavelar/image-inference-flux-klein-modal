# Flux Klein Inference API

API FastAPI autossuficiente para gerar imagens com **Flux Klein 9B**. O runtime do ComfyUI está em [`vendor/comfyui/`](vendor/comfyui/) — copiado do seu ComfyUI local — e usa os modelos deste repositório com **DynamicVRAM** (6 GB VRAM).

## Estrutura

```
image-inference-flux-klein/
├── app/                    # API FastAPI
├── vendor/comfyui/         # Runtime ComfyUI (copiado)
├── diffusion_models/       # flux-2-klein-9b.safetensors
├── text_encoders/          # qwen_3_8b_fp4mixed.safetensors
├── vae/                    # flux2-vae.safetensors
├── loras/                  # LoRAs .safetensors
└── extra_model_paths.yaml  # Mapeia pastas locais para o ComfyUI embarcado
```

## Setup

```bash
cd ~/Projetos/image-inference-flux-klein
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # credenciais AWS já estão no .env
```

## Rodar

```bash
source .venv/bin/activate
python -m app
```

Ou, explicitamente:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

Na primeira request (ou no startup), o runtime ComfyUI carrega os modelos. A inferência leva ~1 minuto em RTX 3050 6 GB.

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Status + `model_loaded` |
| GET | `/loras` | Lista LoRAs em `loras/` |
| POST | `/generate` | Gera imagem e upload S3 |

### Exemplo

```bash
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a portrait of a woman, studio lighting",
    "steps": 6,
    "seed": 42,
    "loras": [
      {"filename": "flux_nina.safetensors", "weight": 1.0}
    ]
  }'
```

Imagens salvas em `inferences/{Ymd}/{uuid}.png` no bucket R2.

## Parâmetros fixos

- 1024×1024, sampler **euler**, scheduler **simple**, cfg **1.0**, FluxGuidance **4.0**

## Atualizar o ComfyUI embarcado

Se atualizar o ComfyUI em `~/Projetos/ComfyUI`, rode:

```bash
bash scripts/sync_comfyui.sh
```
