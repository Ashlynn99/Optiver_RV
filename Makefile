.PHONY: test reports check

test:
	python -m pytest

reports:
	PYTHONPATH=. python scripts/build_report_assets.py

check: test reports
	python -m compileall -q src scripts
