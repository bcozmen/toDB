import React, { useMemo, useState } from 'react';
import { usePatientStore, patientStore } from '../patient/patientStore';
import './ai.css';

function toSeconds(value) {
  if (value == null || value === '') return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : Math.floor(parsed / 1000);
}

function formatDate(seconds) {
  if (seconds == null) return '—';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date(seconds * 1000));
}

function getRiskLevel(val) {
  if (val == null || !Number.isFinite(val)) return { level: 'low', label: 'Low Risk', color: '#71c7aa' };
  if (val >= 0.65) return { level: 'high', label: 'High Risk', color: '#e06666' };
  if (val >= 0.35) return { level: 'moderate', label: 'Moderate Risk', color: '#e6c45c' };
  return { level: 'low', label: 'Low Risk', color: '#71c7aa' };
}

function formatHorizonLabel(label) {
  if (!label) return '';
  return label
    .replace('week', 'wk')
    .replace('month', 'mo')
    .replace('months', 'mos')
    .replace('year', 'yr')
    .replace('years', 'yrs');
}

export default function AI() {
  const { patient, cutoff, aiInsights, aiLoading, aiError } = usePatientStore();
  const [selectedHorizonIdx, setSelectedHorizonIdx] = useState(null);

  // Determine if patient is deceased as of cutoff
  const death = toSeconds(patient?.patient_stop_time);
  const effectiveCutoff = cutoff ?? Math.floor(Date.now() / 1000);
  const isDeceased = death !== null && death <= effectiveCutoff;

  // Process future hazard cumulative distribution
  const chartData = useMemo(() => {
    if (!aiInsights || !Array.isArray(aiInsights.future_hazard) || aiInsights.future_hazard.length === 0) {
      return null;
    }
    const rawHazards = aiInsights.future_hazard;
    const horizons = aiInsights.future_horizon || [];

    const items = rawHazards.map((val, idx) => {
      const clamped = Math.max(0, Math.min(1, Number(val) || 0));
      const horizon = horizons[idx] || `Step ${idx + 1}`;
      const prevVal = idx > 0 ? Math.max(0, Math.min(1, Number(rawHazards[idx - 1]) || 0)) : 0;
      const delta = Math.max(0, clamped - prevVal);
      const risk = getRiskLevel(clamped);

      return {
        index: idx,
        horizon,
        shortHorizon: formatHorizonLabel(horizon),
        cumulativeValue: clamped,
        cumulativePct: Math.round(clamped * 100),
        deltaPct: Math.round(delta * 100),
        risk,
      };
    });

    // Primary 1-year horizon or fallback
    let primaryIdx = items.findIndex((it) => it.horizon.toLowerCase().includes('1 year'));
    if (primaryIdx === -1) {
      primaryIdx = items.findIndex((it) => it.horizon.toLowerCase().includes('6 month'));
    }
    if (primaryIdx === -1) {
      primaryIdx = Math.min(3, items.length - 1);
    }

    return {
      items,
      primaryItem: items[primaryIdx],
    };
  }, [aiInsights]);

  // Chart dimensions & layout
  const chartConfig = useMemo(() => {
    if (!chartData) return null;
    const { items } = chartData;
    const width = 560;
    const height = 180;
    const padLeft = 44;
    const padRight = 32;
    const padTop = 24;
    const padBottom = 34;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const count = items.length;
    const points = items.map((item, idx) => {
      const x = padLeft + (count > 1 ? (idx / (count - 1)) * plotWidth : plotWidth / 2);
      const y = padTop + (1 - item.cumulativeValue) * plotHeight;
      return { ...item, x, y };
    });

    // SVG paths
    const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    const areaPath = [
      `M ${points[0].x.toFixed(1)},${(padTop + plotHeight).toFixed(1)}`,
      ...points.map((p) => `L ${p.x.toFixed(1)},${p.y.toFixed(1)}`),
      `L ${points.at(-1).x.toFixed(1)},${(padTop + plotHeight).toFixed(1)}`,
      'Z',
    ].join(' ');

    const y35 = padTop + (1 - 0.35) * plotHeight;
    const y65 = padTop + (1 - 0.65) * plotHeight;

    return {
      width,
      height,
      padLeft,
      padRight,
      padTop,
      padBottom,
      plotWidth,
      plotHeight,
      points,
      linePath,
      areaPath,
      y35,
      y65,
    };
  }, [chartData]);

  const activeItem = useMemo(() => {
    if (!chartData) return null;
    if (selectedHorizonIdx != null && chartData.items[selectedHorizonIdx]) {
      return chartData.items[selectedHorizonIdx];
    }
    return chartData.primaryItem;
  }, [chartData, selectedHorizonIdx]);

  const handleRefresh = () => {
    patientStore.fetchAiInsights();
  };

  if (!patient) {
    return (
      <section className="ai-panel" aria-label="AI predictive insights">
        <header className="ai-header">
          <div className="ai-header-left">
            <span className="ai-kicker">AI PREDICTIVE INSIGHTS</span>
            <div className="ai-title-row">
              <h2>Future Encounter Risk</h2>
            </div>
          </div>
        </header>
        <div className="ai-state-center">
          <span className="ai-state-icon">🧠</span>
          <h3 className="ai-state-title">No Patient Selected</h3>
          <p className="ai-state-desc">Select or load a patient record to view causal transformer hazard forecasts.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="ai-panel" aria-label="AI predictive insights">
      {/* Header */}
      <header className="ai-header">
        <div className="ai-header-left">
          <span className="ai-kicker">AI PREDICTIVE INSIGHTS</span>
          <div className="ai-title-row">
            <h2>Future Encounter Risk</h2>
            <span className="ai-dot-sep">·</span>
            <span className="ai-subtitle">Cumulative Hazard</span>
            <span className="ai-cutoff-tag">As of {formatDate(effectiveCutoff)}</span>
          </div>
        </div>

        <div className="ai-header-right">
          <button
            type="button"
            className="ai-btn-action"
            onClick={handleRefresh}
            disabled={aiLoading || isDeceased}
            title="Recompute future hazard forecast"
          >
            <span className={`ai-btn-icon ${aiLoading ? 'is-spinning' : ''}`}>↻</span>
            <span>{aiLoading ? 'Forecasting…' : 'Re-run'}</span>
          </button>
        </div>
      </header>

      {/* Main Body */}
      <div className="ai-content">
        {isDeceased ? (
          <div className="ai-deceased-banner">
            <span className="ai-deceased-icon">ℹ️</span>
            <div className="ai-deceased-text">
              <h4 className="ai-deceased-title">Patient is Deceased as of Selected Cut-Off</h4>
              <p className="ai-deceased-detail">
                Patient passed away on {formatDate(death)}. Future clinical visit forecasts and encounter hazards are
                not active after the date of death.
              </p>
            </div>
          </div>
        ) : aiLoading && !chartData ? (
          <div className="ai-state-center">
            <div className="ai-spinner" />
            <h3 className="ai-state-title">Computing Future Hazards</h3>
            <p className="ai-state-desc">Estimating discrete-time survival hazards across clinical horizons…</p>
          </div>
        ) : aiError && !chartData ? (
          <div className="ai-state-center">
            <span className="ai-state-icon">⚠️</span>
            <h3 className="ai-state-title">Forecast Unavailable</h3>
            <p className="ai-state-desc">{aiError}</p>
            <button type="button" className="ai-btn-action" onClick={handleRefresh}>
              Retry Prediction
            </button>
          </div>
        ) : chartData && chartConfig ? (
          <>
            {/* Primary Risk Summary Card */}
            <div className="ai-summary-card">
              <div className="ai-summary-left">
                <div className={`ai-risk-meter meter-${chartData.primaryItem.risk.level}`}>
                  <span className="ai-meter-value">{chartData.primaryItem.cumulativePct}</span>
                  <span className="ai-meter-unit">%</span>
                </div>
                <div className="ai-summary-meta">
                  <span className="ai-summary-label">
                    {chartData.primaryItem.horizon} Cumulative Visit Risk
                  </span>
                  <h3 className="ai-summary-headline">
                    {chartData.primaryItem.risk.label} ({chartData.primaryItem.cumulativePct}% probability)
                  </h3>
                </div>
              </div>

              <div className="ai-summary-right">
                <span className={`ai-risk-badge badge-${chartData.primaryItem.risk.level}`}>
                  <span className="ai-risk-badge-dot" />
                  {chartData.primaryItem.risk.label}
                </span>
              </div>
            </div>

            {/* Interactive Cumulative Hazard Curve Card */}
            <div className="ai-chart-card">
              <div className="ai-card-header">
                <h4 className="ai-card-title">Cumulative Hazard Distribution $H(t)$</h4>
                <div className="ai-chart-legend">
                  <div className="ai-legend-item">
                    <span className="ai-legend-line line-primary" />
                    <span>Predicted Hazard</span>
                  </div>
                  <div className="ai-legend-item">
                    <span className="ai-legend-line line-thresh-mod" />
                    <span>Mod (≥35%)</span>
                  </div>
                  <div className="ai-legend-item">
                    <span className="ai-legend-line line-thresh-high" />
                    <span>High (≥65%)</span>
                  </div>
                </div>
              </div>

              {/* Responsive SVG Chart */}
              <div className="ai-svg-wrap">
                <svg
                  className="ai-hazard-svg"
                  viewBox={`0 0 ${chartConfig.width} ${chartConfig.height}`}
                  preserveAspectRatio="none"
                >
                  <defs>
                    <linearGradient id="aiHazardGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#71c7aa" stopOpacity="0.32" />
                      <stop offset="60%" stopColor="#71c7aa" stopOpacity="0.08" />
                      <stop offset="100%" stopColor="#71c7aa" stopOpacity="0.00" />
                    </linearGradient>
                  </defs>

                  {/* Horizontal Grid Lines */}
                  {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
                    const y = chartConfig.padTop + (1 - frac) * chartConfig.plotHeight;
                    return (
                      <g key={frac}>
                        <line
                          className="ai-grid-line"
                          x1={chartConfig.padLeft}
                          y1={y}
                          x2={chartConfig.width - chartConfig.padRight}
                          y2={y}
                        />
                        <text
                          className="ai-axis-label"
                          x={chartConfig.padLeft - 6}
                          y={y + 3}
                          textAnchor="end"
                        >
                          {Math.round(frac * 100)}%
                        </text>
                      </g>
                    );
                  })}

                  {/* Moderate Threshold Line (35%) */}
                  <line
                    className="ai-threshold-line"
                    stroke="#e6c45c"
                    x1={chartConfig.padLeft}
                    y1={chartConfig.y35}
                    x2={chartConfig.width - chartConfig.padRight}
                    y2={chartConfig.y35}
                  />

                  {/* High Threshold Line (65%) */}
                  <line
                    className="ai-threshold-line"
                    stroke="#e06666"
                    x1={chartConfig.padLeft}
                    y1={chartConfig.y65}
                    x2={chartConfig.width - chartConfig.padRight}
                    y2={chartConfig.y65}
                  />

                  {/* Area fill under curve */}
                  <path d={chartConfig.areaPath} fill="url(#aiHazardGrad)" />

                  {/* Main Curve line */}
                  <path
                    d={chartConfig.linePath}
                    fill="none"
                    stroke="#71c7aa"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />

                  {/* Data Point Markers */}
                  {chartConfig.points.map((pt) => {
                    const isSelected = selectedHorizonIdx === pt.index;
                    return (
                      <g key={pt.horizon}>
                        <circle
                          className={`ai-point-marker ${isSelected ? 'is-active' : ''}`}
                          cx={pt.x}
                          cy={pt.y}
                          r={isSelected ? 6 : 4.5}
                          fill="#141f27"
                          stroke={pt.risk.color}
                          strokeWidth={isSelected ? 3 : 2}
                          onClick={() => setSelectedHorizonIdx(pt.index)}
                        />
                        <text
                          className={`ai-horizon-label ${isSelected ? 'is-hovered' : ''}`}
                          x={pt.x}
                          y={chartConfig.height - 10}
                          onClick={() => setSelectedHorizonIdx(pt.index)}
                        >
                          {pt.shortHorizon}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              {/* Callout of active or hovered horizon */}
              {activeItem && (
                <div className="ai-active-callout">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="ai-callout-horizon">Horizon: {activeItem.horizon}</span>
                    <span className="ai-dot-sep">·</span>
                    <span className="ai-callout-inc">
                      {activeItem.index === 0
                        ? 'Initial hazard window'
                        : `+${activeItem.deltaPct}% interval increase`}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="ai-callout-val" style={{ color: activeItem.risk.color }}>
                      {activeItem.cumulativePct}% Cumulative
                    </span>
                    <span className={`ai-h-tier tier-${activeItem.risk.level}`}>
                      {activeItem.risk.label}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Step-by-Step Horizon Breakdown Cards */}
            <div className="ai-horizon-grid">
              {chartData.items.map((item) => {
                const isSelected = selectedHorizonIdx === item.index;
                return (
                  <div
                    key={item.horizon}
                    className={`ai-horizon-card ${isSelected ? 'is-active' : ''}`}
                    onClick={() => setSelectedHorizonIdx(item.index)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        setSelectedHorizonIdx(item.index);
                      }
                    }}
                  >
                    <div className="ai-h-card-top">
                      <span className="ai-h-name">{item.horizon}</span>
                      <span className={`ai-h-tier tier-${item.risk.level}`}>
                        {item.risk.level}
                      </span>
                    </div>

                    <div className="ai-h-val-row">
                      <span className="ai-h-val">{item.cumulativePct}%</span>
                      {item.index > 0 && (
                        <span className="ai-h-delta">+{item.deltaPct}%</span>
                      )}
                    </div>

                    <div className="ai-progress-track">
                      <div
                        className={`ai-progress-fill fill-${item.risk.level}`}
                        style={{ width: `${item.cumulativePct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        ) : null}
      </div>
    </section>
  );
}