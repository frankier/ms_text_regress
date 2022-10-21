poetry install --all-extras --with=dev && \
	poetry run pip install --force-reinstall --pre \
	--extra-index-url https://download.pytorch.org/whl/nightly/cpu \
	torch
