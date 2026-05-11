# SCANIA Component X — Predictive Maintenance

Cost-aware learning for failure prediction in industrial predictive maintenance.

## Repository structure
├── backend/   — FastAPI ML serving API (Python, LightGBM)
├── frontend/  — React thesis results dashboard
└── .github/   — CI/CD pipeline (GitHub Actions → Cloud Run)
## Live services

- API: https://scania-predictive-ml-api-r4nz54bana-lz.a.run.app/docs
- Dashboard: deployed via CI/CD on push to deploy/prod

## Local development

```bash
# Backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```
