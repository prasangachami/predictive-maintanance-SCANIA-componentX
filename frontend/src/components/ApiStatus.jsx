import React, { useEffect, useState } from 'react'
import { API_URL } from '../data/results'

export default function ApiStatus() {
  const [status, setStatus] = useState('checking')
  const [data, setData] = useState(null)

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(d => { setData(d); setStatus('ok') })
      .catch(() => setStatus('error'))
  }, [])

  const dot = status === 'ok'
    ? 'bg-green-500' : status === 'error'
    ? 'bg-red-500' : 'bg-yellow-400 animate-pulse'

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className={`w-2 h-2 rounded-full ${dot}`} />
      {status === 'ok' && data ? (
        <span className="text-gray-600">
          API live · {data.active_model || 'exp1'} · v{data.version}
        </span>
      ) : status === 'error' ? (
        <span className="text-red-600">API unreachable</span>
      ) : (
        <span className="text-gray-400">Checking...</span>
      )}
    </div>
  )
}
