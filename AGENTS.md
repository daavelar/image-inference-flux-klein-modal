# AGENTS.md — Flux Klein Inference

Guia para agentes de código que trabalham neste repositório.

## O que é este projeto

API de inferência de imagens **Flux Klein 9B** (1024×1024) com runtime **ComfyUI embarcado** em `vendor/comfyui/`. A lógica de negócio fica em `app/`; o ComfyUI é dependência de runtime, não código a refatorar.

Dois modos de execução:

| Modo | Entrada | Onde roda |
|------|---------|-----------|
| **Local** | `python -m app` | Máquina local (GPU ~6 GB VRAM com DynamicVRAM) |
| **Modal** | `generate-modal.py` | GPU na nuvem (A100-40GB→RTX-PRO-6000), volume `fluxkleininf` |

A API expõe os mesmos endpoints em ambos os modos. No Modal, `POST /generate` é **assíncrono** por padrão (202 + poll em `/result/{call_id}`).

## Estrutura do repositório

```
app/                      # Código da API e inferência (editar aqui)
  main.py                 # FastAPI local — rotas /health, /loras, /generate
  inference.py            # Orquestra workflow, upload S3, logging
  workflow.py             # Monta o grafo ComfyUI (Flux Klein)
  comfy_runtime.py        # Bootstrap e execução do ComfyUI embarcado
  config.py               # Settings (Pydantic) — modelos, AWS, paths
  storage.py              # Upload para S3/R2
  schemas.py              # Request/response Pydantic
  reference_images.py     # Download/validação de imagens de referência
generate-modal.py         # Deploy Modal — worker GPU + ASGI HTTP
vendor/comfyui/           # ComfyUI copiado — evitar edits; usar sync script
diffusion_models/         # UNET (flux-2-klein-9b.safetensors)
text_encoders/            # CLIP (qwen_3_8b_fp4mixed.safetensors)
vae/                      # VAE (flux2-vae.safetensors)
loras/                    # LoRAs .safetensors
extra_model_paths.yaml    # Mapeia pastas locais → ComfyUI
scripts/                  # Upload para Modal volume, sync ComfyUI
llm.txt                   # Documentação de integração HTTP (consumidores)
```

**Não existe** `generate.py` — foi removido para evitar confusão. Local = `python -m app`; Modal = `generate-modal.py`.

## Fluxo de inferência

```
POST /generate
  → app/main.py (local) ou generate-modal.py (Modal HTTP)
  → app/inference.py::generate_image()
      → validate loras
      → save reference_images (URLs → vendor/comfyui/input/)
      → app/workflow.py::build_workflow()
      → app/comfy_runtime.py::run_workflow()
      → app/storage.py::upload_image() → R2/S3
  → GenerateResponse { url, key, seed, duration_seconds, cost }
```

Parâmetros fixos (em `app/config.py`): 1024×1024, sampler `euler`, scheduler `simple`, cfg `1.0`, FluxGuidance `4.0`.

## Onde fazer mudanças comuns

| Objetivo | Arquivo(s) |
|----------|------------|
| Novo endpoint ou comportamento HTTP local | `app/main.py` |
| Payload, validação, resposta da API | `app/schemas.py` |
| Lógica de geração, LoRAs, referências | `app/inference.py` |
| Grafo ComfyUI (nós, conexões) | `app/workflow.py` |
| Paths de modelos, env vars, defaults | `app/config.py`, `.env.example` |
| Upload / URL pública da imagem | `app/storage.py` |
| Bootstrap ComfyUI, paths do volume | `app/comfy_runtime.py` |
| Deploy Modal, GPU, volume, HTTP async | `generate-modal.py` |
| Paths dos modelos no ComfyUI | `extra_model_paths.yaml` |
| Docs para integradores externos | `llm.txt` |

## Variáveis de ambiente

Copiar `.env.example` → `.env` para execução local.

Obrigatórias para upload:
- `AWS_BUCKET`, `AWS_ENDPOINT`, `AWS_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

Modal usa **Modal Secret** (nome em `MODAL_AWS_SECRET`) com as mesmas chaves AWS.

Modal também define em runtime:
- `NETWORK_VOLUME_PATH`, `MODELS_BASE`, `REQUIRE_NETWORK_VOLUME=1`

## Comandos úteis

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Local
python -m app                    # porta 8002 (configurável em Settings)
make run                         # supervisord (supervisor.conf)

# Modal
.venv/bin/modal deploy generate-modal.py
.venv/bin/modal run generate-modal.py --payload-json '{"input":{"prompt":"a portrait"}}'

# Popular volume Modal (uma vez)
python scripts/upload-models-to-modal-volume.py
python scripts/upload-loras-to-modal-volume.py

# Atualizar ComfyUI embarcado
bash scripts/sync_comfyui.sh
```

## Deploy Modal

- App: `flux-klein-modal`
- Volume padrão: `fluxkleininf` montado em `/my_vol`
- Modelos esperados: `/my_vol/models/{diffusion_models,text_encoders,vae,loras}`
- GPU padrão: cadeia `A100-40GB` → `A100-80GB` → `L40S` → `RTX-PRO-6000` (`MODAL_GPU` para override)
- Custo por job: detectado no boot via `nvidia-smi` (`app/gpu_pricing.py`); taxa aplicada em `COST_PER_SECOND`
- HTTP label: `flux-http` (URL no output do deploy)

Após alterações em `app/` ou `generate-modal.py`, redeploy:

```bash
.venv/bin/modal deploy generate-modal.py
```

## Convenções para agentes

1. **Escopo mínimo** — Preferir mudanças em `app/`; não editar `vendor/comfyui/` salvo sync ou patch pontual documentado.
2. **Código em inglês** — Identificadores, comentários e strings de log em inglês.
3. **Reutilizar caminhos existentes** — Local usa pastas na raiz do repo; Modal usa `MODELS_BASE` no volume.
4. **Dois entrypoints HTTP** — `app/main.py` (sync) e `generate-modal.py` (async por padrão). Ao mudar contratos da API, atualizar ambos se necessário.
5. **Modelos pesados** — Não commitar `.safetensors`; pastas têm placeholders. Upload via scripts Modal.
6. **Sem testes automatizados** — Não adicionar suite de testes unless pedido; validar com curl/health quando relevante.
7. **Não rodar** `npm run build` nem `php artisan test`.
8. **Commits** — Só quando o usuário pedir explicitamente.

## Endpoints (resumo)

| Método | Rota | Local | Modal |
|--------|------|-------|-------|
| GET | `/health` | `model_loaded` | `model_ready` |
| GET | `/loras` | lista LoRAs | lista LoRAs |
| POST | `/generate` | 200 sync | 202 async (ou `?blocking=true`) |
| GET | `/result/{call_id}` | — | poll do job |

Payload `POST /generate`:

```json
{
  "prompt": "a portrait of a woman, studio lighting",
  "steps": 4,
  "seed": 42,
  "loras": [{"filename": "flux_nina.safetensors", "weight": 1.0}],
  "reference_images": ["https://example.com/ref.png"]
}
```

Imagens salvas em `modal_flux_infs/{YYYYMMDD}/{uuid}.png` no bucket configurado.

## Referências

- Integração HTTP detalhada: `llm.txt`
- Setup humano: `README.md`
- Modal docs: https://modal.com/docs/guide
