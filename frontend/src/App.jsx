import React, { useState } from 'react'
import NavBar from './components/NavBar'
import ApiStatus from './components/ApiStatus'
import Overview from './pages/Overview'
import Experiments from './pages/Experiments'
import CostMatrix from './pages/CostMatrix'
import Discussion from './pages/Discussion'

const pages = {
  overview:    <Overview />,
  experiments: <Experiments />,
  costmatrix:  <CostMatrix />,
  discussion:  <Discussion />,
}

export default function App() {
  const [page, setPage] = useState('overview')

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <div className="text-xs text-gray-400 mb-0.5 uppercase tracking-wide">SCANIA Component X</div>
            <h1 className="text-base font-medium">Predictive maintenance — thesis results</h1>
          </div>
          <ApiStatus />
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6">
        <NavBar active={page} onChange={setPage} />
        {pages[page]}
      </main>

      <footer className="max-w-6xl mx-auto px-6 py-8 mt-8 border-t border-gray-200">
        <p className="text-xs text-gray-400 text-center">
          SCANIA Component X Predictive Maintenance · Cost-aware learning thesis ·
          Deployed on Google Cloud Run · europe-north1
        </p>
      </footer>
    </div>
  )
}
