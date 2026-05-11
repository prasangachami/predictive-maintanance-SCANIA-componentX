import React from 'react'
import { COST_MATRIX, CLASS_LABELS } from '../data/results'

function cellStyle(value) {
  if (value === 0) return 'bg-green-100 text-green-800 font-medium'
  if (value <= 10) return 'bg-blue-50 text-blue-800'
  if (value <= 200) return 'bg-amber-100 text-amber-800'
  if (value <= 300) return 'bg-orange-200 text-orange-900'
  if (value <= 400) return 'bg-red-300 text-red-900'
  return 'bg-red-600 text-white font-medium'
}

const shortLabel = ['0 — healthy', '1 — 24–48 steps', '2 — 12–24 steps', '3 — 6–12 steps', '4 — imminent']

export default function CostMatrix() {
  return (
    <div className="space-y-4">
      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h3 className="text-sm font-medium mb-1">Industrial cost matrix</h3>
        <p className="text-xs text-gray-500 mb-4">
          Rows = actual temporal class · Columns = predicted class · Values = maintenance cost penalty
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr>
                <th className="text-left text-xs text-gray-500 py-2 pr-4 font-medium w-44">Actual \ Predicted</th>
                {shortLabel.map(l => (
                  <th key={l} className="text-center text-xs text-gray-500 py-2 px-3 font-medium">{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {COST_MATRIX.map((row, i) => (
                <tr key={i}>
                  <td className="py-2 pr-4 text-xs text-gray-600 font-medium">{shortLabel[i]}</td>
                  {row.map((val, j) => (
                    <td key={j} className={`text-center py-2.5 px-3 text-sm rounded mx-0.5 ${cellStyle(val)}`}>
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex gap-3 mt-4 flex-wrap">
          {[['bg-green-100 text-green-800','0 — correct prediction'],['bg-amber-100 text-amber-800','7–200 — moderate cost'],['bg-orange-200 text-orange-900','200–300 — high cost'],['bg-red-600 text-white','400–500 — critical FN cost']].map(([cls,label]) => (
            <span key={label} className={`text-xs px-2 py-1 rounded ${cls}`}>{label}</span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-sm font-medium mb-3">Key insight — asymmetric costs</h3>
          <p className="text-sm text-gray-600 leading-relaxed">
            A false negative (predicting healthy when failure is imminent) costs
            <span className="font-medium text-red-700"> 500 units</span>.
            A false positive (predicting failure when healthy) costs only
            <span className="font-medium text-blue-700"> 10 units</span>.
            Standard cross-entropy treats these symmetrically — cost-aware focal loss
            corrects this by weighting the FN penalty directly into the loss function.
          </p>
        </div>
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-sm font-medium mb-3">Two label systems</h3>
          <div className="space-y-2 text-sm text-gray-600">
            <div className="flex gap-2 items-start">
              <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-800 mt-0.5 shrink-0">Binary</span>
              <span>Labels {'{'}0,1{'}'} used for model training only. Threshold optimised per experiment.</span>
            </div>
            <div className="flex gap-2 items-start">
              <span className="text-xs px-2 py-0.5 rounded bg-amber-100 text-amber-800 mt-0.5 shrink-0">5-class</span>
              <span>Temporal class {'{'}0–4{'}'} used for cost matrix evaluation only. Never mixed with binary training.</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h3 className="text-sm font-medium mb-3">Class definitions</h3>
        <div className="grid grid-cols-5 gap-3">
          {CLASS_LABELS.map((label, i) => (
            <div key={i} className={`rounded-lg p-3 text-center ${i === 0 ? 'bg-green-50' : i === 4 ? 'bg-red-50' : 'bg-gray-50'}`}>
              <div className={`text-lg font-medium mb-1 ${i === 0 ? 'text-green-700' : i === 4 ? 'text-red-700' : 'text-gray-700'}`}>{i}</div>
              <div className="text-xs text-gray-500">{label.split('—')[1]?.trim()}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
