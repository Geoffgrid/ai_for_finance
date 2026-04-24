# MODULE INSTALLS
CPU ?= 1
DATA_PERIOD_YEARS ?= 3
DATA_DELTA_DAYS ?= 365
DATA_TICKERS ?= BTC-USD,ETH-USD,SPY

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


build_img_local:
	docker build -t ${IMAGE}:local .


build_for_production:
	docker build \
		--platform linux/amd64 \
    -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.

push_image_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACTSREPO}/${IMAGE}:prod

deploy_to_cloud_run:
	gcloud run deploy \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACTSREPO}/${IMAGE}:prod \
		--memory ${MEMORY} \
		--cpu ${CPU} \
		--set-env-vars "^@^DATA_TICKERS=${DATA_TICKERS}@DATA_PERIOD_YEARS=${DATA_PERIOD_YEARS}@DATA_DELTA_DAYS=${DATA_DELTA_DAYS}" \
		--region ${GCP_REGION}

run_streamlit:
	streamlit run app/frontend_file.py

# GCP DEPLOY HELPERS (added, existing targets kept intact)
gcp_auth:
	gcloud auth login
	gcloud auth application-default login

gcp_set_project:
	gcloud config set project ${GCP_PROJECT_ID}

gcp_enable_services:
	gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

gcp_configure_docker_auth:
	gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

gcp_create_artifact_repo:
	gcloud artifacts repositories create ${ARTIFACTSREPO} \
		--repository-format=docker \
		--location=${GCP_REGION} \
		--description="Docker repository for ai_for_finance" || :

gcp_check_vars:
	@test -n "${GCP_PROJECT_ID}" || (echo "GCP_PROJECT_ID is required" && exit 1)
	@test -n "${GCP_REGION}" || (echo "GCP_REGION is required" && exit 1)
	@test -n "${ARTIFACTSREPO}" || (echo "ARTIFACTSREPO is required" && exit 1)
	@test -n "${IMAGE}" || (echo "IMAGE is required" && exit 1)
	@test -n "${MEMORY}" || (echo "MEMORY is required" && exit 1)
	@test -n "${CPU}" || (echo "CPU is required (example: CPU=1)" && exit 1)

gcp_setup: gcp_check_vars gcp_set_project gcp_enable_services gcp_configure_docker_auth gcp_create_artifact_repo

deploy_prod: gcp_setup build_for_production push_image_production deploy_to_cloud_run
