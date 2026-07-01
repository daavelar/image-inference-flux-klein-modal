#!/usr/bin/env bash
# Corrige o loop de crash da API. Dois problemas encontrados nos logs:
#   1. torchvision 0.27.0 (build generico) incompativel com torch 2.11.0+cu128
#      -> RuntimeError: operator torchvision::nms does not exist
#   2. dependencias faltando no venv: transformers, spandrel
#      -> ModuleNotFoundError: No module named 'transformers'
#
# Abordagem: manter o torch 2.11.0+cu128 ja instalado e casar os pares cu128.
# NAO use "make install": o requirements.txt tem "torchvision" sem versao fixa e
# baixaria de novo o build generico quebrado (0.27.0), desfazendo a correcao.
set -euo pipefail

cd /workspace
PIP=.venv/bin/pip
CU128="https://download.pytorch.org/whl/cu128"

echo ">> [1/3] Casando torchvision/torchaudio com torch 2.11.0+cu128"
$PIP install --no-deps \
  "torchvision==0.26.0+cu128" \
  "torchaudio==2.11.0+cu128" \
  --index-url "$CU128"

echo ">> [2/3] Instalando dependencias faltando (sem mexer no torch)"
$PIP install --no-deps spandrel
$PIP install "transformers>=4.50.3"

echo ">> [3/3] Reiniciando supervisord (limpa instancias duplicadas) e a API"
pkill -f supervisord 2>/dev/null || true
pkill -f '.venv/bin/python -m app' 2>/dev/null || true
sleep 2
make run

echo
echo ">> Pronto. Acompanhe com: make logs"
echo ">> Esperado: 'Application startup complete' no lugar do traceback."
