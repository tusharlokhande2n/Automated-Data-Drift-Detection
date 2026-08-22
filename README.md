# Data Drift Monitor

A Streamlit application that compares baseline and production CSV data, identifies feature drift, monitors model performance, creates audit downloads, and can send Gmail alerts.

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

For Gmail alerts, set `DRIFT_ALERT_SENDER` and `DRIFT_ALERT_APP_PASSWORD` in the deployment environment. Use a Gmail app password, never an account password.

## CI/CD

GitHub Actions automatically runs syntax checks, unit tests, and linting for pull requests and pushes. A successful push to `main` publishes a Docker image to GitHub Container Registry and triggers Render when the `RENDER_DEPLOY_HOOK_URL` secret is present.

Before the first production deployment:

1. Create a GitHub repository and push this project to its `main` branch.
2. Create a Docker-based Render web service from this repository, or use the included `render.yaml` blueprint.
3. Add `RENDER_DEPLOY_HOOK_URL` as a GitHub Actions secret.
4. Add `DRIFT_ALERT_SENDER` and `DRIFT_ALERT_APP_PASSWORD` as Render environment secrets.
