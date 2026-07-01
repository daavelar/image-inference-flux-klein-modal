.PHONY: run install logs

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

run: $(PYTHON)
	@mkdir -p logs
	supervisord -c supervisor.conf

install: $(PYTHON)
	$(PIP) install -r requirements.txt

logs:
	tail -f logs/api.log

$(PYTHON):
	@test -x $(PYTHON) || (echo "Virtualenv not found. Run: python3 -m venv .venv && make install" && exit 1)
