poetry install --all-extras --with=dev && \
	poetry run pip install --force-reinstall \
	transformers[torch] torch --index-url https://download.pytorch.org/whl/cpu
