import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { EXPERIMENTS } from '../data/results'

const costData = EXPERIMENTS.map(e => ({
  name: e.name, cost: e.testCost, fill: e.color
}))

const aucData = EXPERIMENTS.map(e => ({
  name: e.name, validation: e.valAuc, test: e.testAuc
}))

const prfData = EXPERIMENTS.map(e => ({
  name: e.name,
  precision: parseFloat((e.precision * 100).toFixed(2)),
  recall: parseFloat((e.recall * 100).toFixed(2)),
  f1: parseFloat((e.f1 * 100).toFixed(2)),
}))

function CustomBar(props) {
  const { fill, x, y, width, height, name } = props
  const exp = EXPERIMENTS.find(e => e.name === name)
  return <rect x={x} y={y} width={width} height={height}
    fill={exp ? exp.color : fill} rx={3} />
}

export default function Experiments() {
  return (
    <div className="space-y-4">
      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h3 className="text-sm font-medium mb-1">Test total cost — all experiments</h3>
        <p className="text-xs text-gray-400 mb-4">Lower is better — cost calculated using the industrial 5×5 cost matrix</p>
        <div className="flex gap-4 flex-wrap mb-3">
          {EXPERIMENTS.map(e => (
            <span key={e.id} className="flex items-center gap-1.5 text-xs text-gray-500">
              <span className="w-2.5 h-2.5 rounded-sm" style={{ background: e.color }} />
              {e.name}
            </span>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={costData} margin={{ top: 4, right: 8, left: 8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={v => v.toLocaleString()} />
            <Tooltip formatter={v => [v.toLocaleString(), 'Total cost']} />
            <Bar dataKey="cost" shape={<CustomBar />} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-sm font-medium mb-1">AUC-ROC — validation vs test</h3>
          <div className="flex gap-4 mb-3">
            <span className="flex items-center gap-1.5 text-xs text-gray-500">
              <span className="w-2.5 h-2.5 rounded-sm bg-gray-400" />Validation
            </span>
            <span className="flex items-center gap-1.5 text-xs text-gray-500">
              <span className="w-2.5 h-2.5 rounded-sm bg-purple-600" />Test
            </span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={aucData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0.45, 0.72]} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(2)} />
              <Tooltip formatter={v => v.toFixed(4)} />
              <Bar dataKey="validation" fill="#888780" radius={[3,3,0,0]} />
              <Bar dataKey="test" fill="#534AB7" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-sm font-medium mb-1">Precision · recall · F1 (%)</h3>
          <div className="flex gap-3 mb-3">
            {[['#185FA5','Precision'],['#1D9E75','Recall'],['#EF9F27','F1']].map(([c,l]) => (
              <span key={l} className="flex items-center gap-1.5 text-xs text-gray-500">
                <span className="w-2.5 h-2.5 rounded-sm" style={{ background: c }} />{l}
              </span>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={prfData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={v => v + '%'} />
              <Tooltip formatter={v => v.toFixed(2) + '%'} />
              <Bar dataKey="precision" fill="#185FA5" radius={[3,3,0,0]} />
              <Bar dataKey="recall" fill="#1D9E75" radius={[3,3,0,0]} />
              <Bar dataKey="f1" fill="#EF9F27" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h3 className="text-sm font-medium mb-4">Full metrics table</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-200">
                <th className="text-left py-2 pr-4">Experiment</th>
                <th className="text-left py-2 pr-4">Loss function</th>
                <th className="text-right py-2 pr-4">Val AUC</th>
                <th className="text-right py-2 pr-4">Test AUC</th>
                <th className="text-right py-2 pr-4">Precision</th>
                <th className="text-right py-2 pr-4">Recall</th>
                <th className="text-right py-2 pr-4">F1</th>
                <th className="text-right py-2 pr-4">Test cost</th>
                <th className="text-right py-2">Cost reduction</th>
              </tr>
            </thead>
            <tbody>
              {EXPERIMENTS.map(e => (
                <tr key={e.id} className={`border-b border-gray-50 ${e.isBest ? 'bg-green-50' : ''}`}>
                  <td className="py-2.5 pr-4 font-medium">{e.name}</td>
                  <td className="py-2.5 pr-4 text-gray-500 text-xs">{e.loss}</td>
                  <td className="py-2.5 pr-4 text-right">{e.valAuc.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 text-right font-medium">{e.testAuc.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 text-right">{e.precision.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 text-right">{e.recall.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 text-right">{e.f1.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 text-right">{e.testCost.toLocaleString()}</td>
                  <td className="py-2.5 text-right">
                    {e.costReduction > 0
                      ? <span className="text-xs px-2 py-0.5 rounded bg-green-100 text-green-800 font-medium">{e.costReduction.toFixed(1)}%</span>
                      : <span className="text-xs text-gray-400">baseline</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
