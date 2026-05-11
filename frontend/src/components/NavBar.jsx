import React from 'react'

const tabs = [
  { id: 'overview',     label: 'Overview' },
  { id: 'experiments',  label: 'Experiment comparison' },
  { id: 'costmatrix',   label: 'Cost matrix' },
  { id: 'discussion',   label: 'Results & discussion' },
]

export default function NavBar({ active, onChange }) {
  return (
    <nav className="border-b border-gray-200 mb-8">
      <div className="flex gap-0">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => onChange(t.id)}
            className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
              active === t.id
                ? 'border-gray-900 text-gray-900'
                : 'border-transparent text-gray-500 hover:text-gray-900'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
    </nav>
  )
}
