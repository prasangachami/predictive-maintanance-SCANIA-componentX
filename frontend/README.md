# Frontend — Interactive Results Dashboard

An interactive dashboard built with React, Vite, and Tailwind CSS that visualises the full experimental results of the cost-aware predictive maintenance study.

---

## Live Dashboard

> **[(https://scania-dashboard-185324232016.europe-north1.run.app/)]**

---

## Structure

```
frontend/
├── src/
│   ├── App.jsx                 ← root app with routing
│   ├── main.jsx                ← entry point
│   ├── index.css               ← global styles
│   │
│   ├── pages/                  ← one component per dashboard page
│   │   ├── Overview.jsx        ← experiment comparison and summary
│   │   ├── Experiments.jsx     ← detailed per-experiment results
│   │   ├── CostMatrix.jsx      ← cost matrix visualisation
│   │   └── Discussion.jsx      ← key findings and research questions
│   │
│   ├── components/             ← reusable UI components
│   │   ├── NavBar.jsx          ← navigation bar
│   │   ├── MetricCard.jsx      ← metric display card
│   │   └── ApiStatus.jsx       ← backend API connection status
│   │
│   └── data/
│       └── results.js          ← pre-computed experiment results
│
├── public/                     ← static assets
├── index.html                  ← HTML entry point
├── nginx.conf                  ← production nginx configuration
├── Dockerfile                  ← containerised frontend
├── package.json                ← Node.js dependencies
├── vite.config.js              ← Vite build configuration
├── tailwind.config.js          ← Tailwind CSS configuration
└── postcss.config.js           ← PostCSS configuration
```

---

## Pages

**Overview** — side-by-side comparison of all four experiments. Shows cost, AUC-ROC, recall, precision, and F1-score in a single view. The central trade-off — Exp 3 best AUC vs Exp 4 lowest cost — is immediately visible.

**Experiments** — detailed breakdown per experiment. Includes threshold configuration (t1–t4), cost breakdown (FN vs FP), and classification metrics. Explains why Exp 3 increased cost despite perfect recall, and why Exp 4 is a degenerate result.

**Cost Matrix** — interactive visualisation of the 5×5 industrial cost matrix. Shows how false-negative costs (200–500 units) compare to false-positive costs (7–10 units) and how this asymmetry drives the training objective.

**Discussion** — answers to the three research questions (RQ1, RQ2, RQ3) and the sensitivity analysis results (fn_weight sweep 0.25 → 3.0).

---

## Run Locally

### Prerequisites

- Node.js 18 or higher
- npm 9 or higher

### Install and start

```bash
cd frontend
npm install
npm run dev
```

The dashboard opens at `http://localhost:5173`.

### Build for production

```bash
npm run build
```

Built files are output to `frontend/dist/`. The `dist/` folder is gitignored — it is generated at deploy time, not committed to the repository.

---

## Docker

```bash
cd frontend
docker build -t scania-frontend .
docker run -p 80:80 scania-frontend
```

The container serves the built dashboard via nginx on port 80.

---

## Data Source

The dashboard reads from `src/data/results.js` — a pre-computed JavaScript module containing all confirmed experiment results. This means:

- The dashboard works without the dataset
- The dashboard works without the backend API running
- Anyone can view results without installing Python or running experiments

The `ApiStatus` component shows whether the backend FastAPI is reachable. If the backend is offline, all static result data still displays correctly.

To update the results data after re-running experiments, export from the backend notebook:

```bash
# Run backend/notebooks/06_merge_results.ipynb
# Then update frontend/src/data/results.js with the new values
```

---

## Backend API Connection

The dashboard can connect to the FastAPI backend for live predictions. The backend URL is configured in `vite.config.js`:

```js
// vite.config.js
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

Start the backend first if you want live API features:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The `ApiStatus` component in the navbar shows the connection status.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| React 18 | UI framework |
| Vite | Build tool and dev server |
| Tailwind CSS | Utility-first styling |
| nginx | Production web server |
| Docker | Containerisation |

---

## Deployment

The frontend is configured for deployment on any static hosting platform or container service. The `nginx.conf` handles client-side routing for the single-page app.

**Recommended platforms:**

- **Vercel** — connect GitHub repo, set build command `npm run build`, output dir `dist`
- **Netlify** — same as Vercel
- **Google Cloud Run** — use the provided Dockerfile with Cloud Build (`cloudbuild.yaml` at project root)
- **GitHub Pages** — run `npm run build` and deploy the `dist/` folder
