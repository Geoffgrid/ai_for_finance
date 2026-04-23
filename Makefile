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
		--region ${GCP_REGION}

run_streamlit:
	streamlit run app/frontend_file.py

deploy_to_cloud_run_poste_geoffroy:
	gcloud run deploy my-api-app \                                                                                                                                                                                    [🐍 ai_for_finance]
  --image europe-west1-docker.pkg.dev/ai-for-finance-494120/my-artifact-repo/my-api-app:prod \
  --memory 1Gi \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --timeout 300 \
  --cpu 1
