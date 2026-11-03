# Image Inference Flux Klein — Modal

Endpoint [Modal](https://modal.com) para geração e edição de imagens com **Flux.2 Klein 9B**. O worker sobe um runtime ComfyUI na GPU, aplica LoRAs opcionais, faz upload da imagem para Cloudflare R2 e devolve a URL pública.

## Como funciona

```
Cliente → POST /generate (Modal HTTP)
       → FluxKleinWorker carrega UNET + CLIP + VAE
       → grafo ComfyUI (texto, ou texto + imagens de referência)
       → upload R2
       → resposta { url, key, seed, duration_seconds, cost }
```

Os pesos ficam num Volume Modal. O container guarda o runtime entre requisições.

## Estrutura do projeto

```
.
├── generate-modal.py       # Entrypoint Modal (worker GPU + HTTP)
├── app/
│   ├── main.py             # FastAPI local (dev)
│   ├── inference.py        # Geração + upload
│   ├── workflow.py         # Grafo ComfyUI (API format)
│   ├── comfy_runtime.py    # Sobe o ComfyUI embarcado
│   ├── reference_images.py # URL ou base64 → arquivo de referência
│   ├── storage.py          # Upload boto3 → R2
│   ├── schemas.py          # Contratos request/response
│   └── config.py           # Settings
├── scripts/
│   ├── upload-models-to-modal-volume.py
│   └── upload-loras-to-modal-volume.py
├── requirements.txt
└── .env.example            # Chaves R2 (copiar para .env, não commitar)
```

Pesos (`.safetensors`) não entram no git. No deploy eles vivem no volume.

## Pré-requisitos

1. Conta Modal: `pip install modal && modal setup`
2. Volume Modal `fluxkleininf` com os pesos
3. Secret Modal com credenciais R2 (`MODAL_AWS_SECRET`)

### Layout no volume Modal

| Caminho no volume | Conteúdo |
|-------------------|----------|
| `/my_vol/models/diffusion_models/flux-2-klein-9b.safetensors` | UNET Klein 9B |
| `/my_vol/models/text_encoders/qwen_3_8b_fp4mixed.safetensors` | CLIP, tipo `flux2` |
| `/my_vol/models/vae/flux2-vae.safetensors` | VAE |
| `/my_vol/models/loras/` | LoRAs `.safetensors` |

### Upload dos pesos

```bash
python scripts/upload-models-to-modal-volume.py
python scripts/upload-loras-to-modal-volume.py
modal volume ls fluxkleininf /models
```

O script espera as pastas locais `models/diffusion_models`, `models/text_encoders`, `models/vae` e `models/loras`. Alternativa: `modal volume put fluxkleininf ./models /models`.

### Secret R2 na Modal

```bash
modal secret create flux-klein-r2 \
  AWS_BUCKET=... \
  AWS_DEFAULT_REGION=auto \
  AWS_ENDPOINT=... \
  AWS_URL=... \
  AWS_ACCESS_KEY_ID=... \
  AWS_SECRET_ACCESS_KEY=... \
  AWS_USE_PATH_STYLE_ENDPOINT=false
```

## Deploy

```bash
export MODAL_AWS_SECRET=flux-klein-r2
modal deploy generate-modal.py
```

URL HTTP: `https://<seu-usuario>--flux-http.modal.run`

App Modal: `flux-klein-modal`.

## API

### Endpoints

| Rota | Descrição |
|------|-----------|
| `GET /health` | Status + model_ready + loras_count |
| `GET /loras` | Lista LoRAs no volume |
| `POST /generate` | Async (202 + call_id) |
| `POST /generate?blocking=true` | Sync — aguarda resultado |
| `GET /result/{call_id}` | Poll do resultado async |

O body pode ser flat ou embrulhado em `input`. `callback_url` e `external_id` podem ir no topo ou dentro de `input`.

### Request

```json
{
  "prompt": "a portrait of a woman, studio lighting",
  "steps": 4,
  "width": 768,
  "height": 1536,
  "seed": 42,
  "loras": [
    { "filename": "example-lora.safetensors", "weight": 1.0 }
  ],
  "reference_images": ["https://example.com/ref.png"],
  "callback_url": "https://your-app.example/api/webhooks/modal/image",
  "external_id": 123
}
```

### Response (sucesso)

```json
{
  "url": "https://.../modal_flux_infs/20260707/uuid.png",
  "key": "modal_flux_infs/20260707/uuid.png",
  "seed": 42,
  "duration_seconds": 18.2,
  "cost": 0.012
}
```

`cost` é uma estimativa em USD a partir do tempo de parede do container (GPU detectada + CPU + memória), não a fatura da Modal.

### Response (async)

```json
{
  "status": "pending",
  "call_id": "...",
  "task_id": "...",
  "result_url": "/result/..."
}
```

### Response (erro)

```json
{ "error": "prompt is required" }
```

## Parâmetros

| Campo | Obrigatório | Padrão | Descrição |
|-------|-------------|--------|-----------|
| `prompt` | sim | — | Prosa em inglês. Klein não usa tags Danbooru |
| `steps` | não | `4` | 1–50. Destilado 9B fica em 4 |
| `seed` | não | aleatório | Inteiro fixo reproduz o resultado |
| `width` / `height` | não | `768` × `1536` | 256–2048, alinhado a múltiplos de 16 |
| `loras` | não | `[]` | `{ "filename": "arquivo.safetensors", "weight": 1.0 }` (0–2) |
| `reference_images` | não | `[]` | Até 4 imagens: URL `http(s)`, data-URI ou base64. PNG, JPEG ou WebP |
| `reference_megapixels` | não | — | 0–4. Orçamento de megapixels das referências |
| `callback_url` | não | — | POST ao concluir, no lugar do polling |
| `external_id` | não | — | Repassado no payload do webhook |

Sampler, scheduler, CFG e FluxGuidance são fixos no grafo: **euler**, **simple**, CFG **1.0**, FluxGuidance **4.0**.

## Teste CLI

```bash
modal run generate-modal.py --payload-json '{"input":{"prompt":"a portrait of a woman, studio lighting"}}'
```

### Exemplo curl (sync)

```bash
curl -X POST "https://<user>--flux-http.modal.run/generate?blocking=true" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a portrait of a woman, studio lighting","steps":4}'
```

### Exemplo curl (async)

```bash
curl -X POST "https://<user>--flux-http.modal.run/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a portrait of a woman, studio lighting"}'

curl "https://<user>--flux-http.modal.run/result/{call_id}"
```

## Desenvolvimento local

Copie `.env.example` para `.env` e preencha as credenciais R2. Coloque os pesos nas pastas que o `.env.example` nomeia (`diffusion_models/`, `text_encoders/`, `vae/`, `loras/`). O runtime ComfyUI não entra no git: `bash scripts/sync_comfyui.sh` copia de `~/Projetos/ComfyUI` para `vendor/comfyui/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app
```

Equivalente: `uvicorn app.main:app --host 0.0.0.0 --port 8002`.

A API local é síncrona: `POST /generate` espera a imagem. Não tem o poll de `call_id` do deploy Modal.

## Variáveis de ambiente

| Variável | Default | Uso |
|----------|---------|-----|
| `MODAL_VOLUME_NAME` | `fluxkleininf` | Nome do volume Modal |
| `MODAL_VOLUME_MOUNT` | `/my_vol` | Ponto de montagem |
| `MODAL_AWS_SECRET` | — | Nome do secret R2 na Modal |
| `MODAL_GPU` | `A100-40GB,A100-80GB,L40S,RTX-PRO-6000` | Fallback de GPU, mais forte primeiro |
| `COST_PER_SECOND` | conforme a GPU detectada | Override manual do custo/segundo |
| `DEFAULT_STEPS` | `4` | Passos quando o request omite `steps` |
| `DIFFUSION_MODEL` | `flux-2-klein-9b.safetensors` | Arquivo do UNET |
| `TEXT_ENCODER` | `qwen_3_8b_fp4mixed.safetensors` | Arquivo do CLIP |
| `VAE` | `flux2-vae.safetensors` | Arquivo do VAE |

### GPUs da cadeia padrão

| GPU | Custo/s (lista Modal usada na estimativa) |
|-----|------------------------------------------|
| Nvidia A100 40 GB | $0.000583 |
| Nvidia A100 80 GB | $0.000694 |
| Nvidia L40S | $0.000542 |
| RTX PRO 6000 | $0.000842 |

A estimativa também soma CPU e memória do container. Ajuste com `MODAL_CPU_CORES`, `MODAL_MEMORY_GIB`, `MODAL_REGION_MULTIPLIER` e `MODAL_TAIL_IDLE_SECONDS`.
