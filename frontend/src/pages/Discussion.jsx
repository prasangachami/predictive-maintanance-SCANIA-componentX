import React from 'react'

const findings = [
  {
    rq: 'RQ1',
    title: 'Does cost-aware training reduce total maintenance cost?',
    answer: 'Yes',
    color: 'green',
    body: 'Exp 3 reduces test total cost from 46,393 to 28,050 — a 39.5% reduction. Standard cross-entropy (exp 1) and focal loss (exp 2) produce nearly identical costs (46,393 vs 46,345), confirming that task-agnostic losses fail to capture the industrial cost asymmetry. Only the cost-aware focal loss with direct FN weighting achieves meaningful cost reduction.',
    badges: ['39.5% cost reduction confirmed', 'RQ1 answered positively'],
    badgeColor: 'green',
  },
  {
    rq: 'RQ2',
    title: 'How does failure-class weight affect the FN/FP trade-off?',
    answer: 'Yes',
    color: 'green',
    body: 'Increasing the FN weight from 0.5 (exp 4) to 1.0 (exp 3) moves the model from zero recall to perfect recall (1.0000). The cost matrix penalises false negatives at 500 units vs 10 units for false positives — so maximising recall at the expense of precision is the correct industrial strategy. Exp 3 optimal threshold of 0.01 reflects this aggressive recall bias.',
    badges: ['Perfect recall at w=1.0', 'FN/FP trade-off confirmed'],
    badgeColor: 'green',
  },
  {
    rq: 'RQ3',
    title: 'How sensitive is performance to the choice of class weights?',
    answer: 'Partial',
    color: 'amber',
    body: 'Results show high sensitivity. At w=0.5 (exp 4) the model collapses to AUC 0.500 and zero recall — essentially predicting the majority class. At w=1.0 (exp 3) it achieves the best AUC (0.6647) and perfect recall. This sharp threshold effect suggests the cost-aware focal loss is sensitive to weight calibration and warrants further sensitivity analysis around w=1.0.',
    badges: ['High sensitivity observed', 'Further sweep needed'],
    badgeColor: 'amber',
  },
]

const limitations = [
  { issue: 'Low precision (0.0281) in exp 3 may cause alert fatigue in production', future: 'Calibrate threshold using full cost matrix optimisation per operating point' },
  { issue: 'Exp 4 model collapse at w=0.5 requires explanation', future: 'Finer-grained sensitivity sweep around w=0.75–1.25' },
  { issue: 'Single dataset (SCANIA APS failure dataset)', future: 'Validate on other industrial failure prediction datasets' },
  { issue: 'No temporal sequence features exploited', future: 'Add LSTM or temporal convolutional model for TTE regression' },
  { issue: 'AUC 0.6647 indicates moderate discriminative ability', future: 'Feature engineering from raw histogram bins, interaction features' },
]

export default function Discussion() {
  return (
    <div className="space-y-4">
      {findings.map(f => (
        <div key={f.rq} className="bg-white border border-gray-200 rounded-xl p-5">
          <div className="flex items-start gap-3 mb-3">
            <span className="text-xs font-medium text-gray-400 mt-0.5 w-8 shrink-0">{f.rq}</span>
            <div className="flex-1">
              <h3 className="text-sm font-medium mb-2">{f.title}</h3>
              <p className="text-sm text-gray-600 leading-relaxed mb-3">{f.body}</p>
              <div className="flex gap-2 flex-wrap">
                {f.badges.map(b => (
                  <span key={b} className={`text-xs px-2 py-0.5 rounded font-medium ${
                    f.badgeColor === 'green' ? 'bg-green-100 text-green-800' : 'bg-amber-100 text-amber-800'
                  }`}>{b}</span>
                ))}
              </div>
            </div>
            <span className={`text-xs px-2 py-0.5 rounded font-medium shrink-0 ${
              f.color === 'green' ? 'bg-green-100 text-green-800' : 'bg-amber-100 text-amber-800'
            }`}>{f.answer}</span>
          </div>
        </div>
      ))}

      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h3 className="text-sm font-medium mb-4">Limitations and future work</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-gray-200">
              <th className="text-left py-2 pr-4">Limitation</th>
              <th className="text-left py-2">Future direction</th>
            </tr>
          </thead>
          <tbody>
            {limitations.map((l, i) => (
              <tr key={i} className="border-b border-gray-50">
                <td className="py-3 pr-4 text-gray-600 align-top">{l.issue}</td>
                <td className="py-3 text-gray-500 align-top">{l.future}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h3 className="text-sm font-medium mb-3">Conclusion</h3>
        <p className="text-sm text-gray-600 leading-relaxed">
          Cost-aware focal loss with FN weight w=1.0 reduces total maintenance cost by 39.5% compared to the cross-entropy baseline on the SCANIA APS failure dataset. The result directly answers RQ1 and demonstrates that incorporating the industrial cost matrix into the training objective is a viable and effective approach for predictive maintenance. The perfect recall achieved by exp 3 is particularly valuable in the industrial context where missed failures carry costs 50× higher than false alarms.
        </p>
      </div>
    </div>
  )
}
