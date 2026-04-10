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
		--region ${GCP_REGION}
