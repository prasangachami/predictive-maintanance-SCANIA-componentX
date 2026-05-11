import React from 'react'
import MetricCard from '../components/MetricCard'
import { EXPERIMENTS } from '../data/results'

const best = EXPERIMENTS.find(e => e.isBest)
const baseline = EXPERIMENTS.find(e => e.isBaseline)

export default function Overview() {
  return (
    <div>
      <div className="mb-8">
        <h2 className="text-lg font-medium mb-1">Project overview</h2>
        <p className="text-sm text-gray-500">
          Cost-aware learning for failure prediction in industrial predictive maintenance.
          Thesis project — Chalmers / KTH · SCANIA Component X dataset.
        </p>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-8">
        <MetricCard label="Best model" value="Exp 3" sub="Cost-aware focal (w=1.0)" />
        <MetricCard label="Best test AUC-ROC" value="0.6647" sub={`vs ${baseline.testAuc.toFixed(4)} baseline`} />
        <MetricCard label="Cost reduction" value="39.5%" sub="Exp 3 vs exp 1 baseline" highlight />
        <MetricCard label="Test total cost" value="28,050" sub={`vs ${baseline.testCost.toLocaleString()} baseline`} />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-sm font-medium mb-4">Research questions</h3>
          <div className="space-y-3">
            {[
              { q: 'RQ1', text: 'Does cost-aware training reduce total cost vs standard loss?', answer: 'Yes', color: 'bg-green-100 text-green-800' },
              { q: 'RQ2', text: 'How does failure-class weight affect FN/FP trade-off?', answer: 'Yes', color: 'bg-green-100 text-green-800' },
              { q: 'RQ3', text: 'How sensitive is performance to class weight choice?', answer: 'Partial', color: 'bg-amber-100 text-amber-800' },
            ].map(r => (
              <div key={r.q} className="flex items-start gap-3">
                <span className="text-xs text-gray-400 font-medium mt-0.5 w-8">{r.q}</span>
                <span className="text-sm text-gray-600 flex-1">{r.text}</span>
                <span className={`text-xs px-2 py-0.5 rounded font-medium ${r.color}`}>{r.answer}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-sm font-medium mb-4">Experiment summary</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-100">
                <th className="text-left pb-2">Exp</th>
                <th className="text-left pb-2">Loss function</th>
                <th className="text-right pb-2">Test cost</th>
                <th className="text-right pb-2">AUC</th>
              </tr>
            </thead>
            <tbody>
              {EXPERIMENTS.map(e => (
                <tr key={e.id} className={`border-b border-gray-50 ${e.isBest ? 'bg-green-50' : ''}`}>
                  <td className="py-2 font-medium">{e.name}</td>
                  <td className="py-2 text-gray-600 text-xs">{e.loss}</td>
                  <td className="py-2 text-right">{e.testCost.toLocaleString()}</td>
                  <td className="py-2 text-right">{e.testAuc.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h3 className="text-sm font-medium mb-3">Deployment</h3>
        <div className="flex items-center gap-2 flex-wrap text-sm">
          {['GitHub (private)', 'GitHub Actions CI/CD', 'Artifact Registry', 'Cloud Run API', 'GCS model registry', 'Vertex AI registry'].map((s, i) => (
            <React.Fragment key={s}>
              <span className={`px-3 py-1.5 rounded-md text-sm ${
                s === 'Cloud Run API' ? 'bg-green-100 text-green-800 font-medium' :
                s === 'GCS model registry' ? 'bg-blue-100 text-blue-800' :
                'bg-gray-100 text-gray-700'
              }`}>{s}</span>
              {i < 5 && <span className="text-gray-400">→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  )
}
