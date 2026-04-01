# MODULE INSTALLS
freeze:
	pip freeze > requirements.txt

install:
	@pip install -r requirements.txt

install_package:
	@pip uninstall -y app || :
	@pip install -e .

# API
run_api:
	uvicorn app.api.fast:app --reload