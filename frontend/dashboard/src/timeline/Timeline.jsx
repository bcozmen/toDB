import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { usePatientStore, patientStore } from '../patient/patientStore';
import { COLORS, EVENT_LABELS } from './timelineConstants';
import './timeline.css';

const DAY = 86400;
const YEAR = 365 * DAY;
const CARD_WIDTH = 210;
const LANE_HEIGHT = 86;
const TRACK_PADDING_X = 48;

function toSeconds(value) {
  if (value == null || value === '') return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : Math.floor(parsed / 1000);
}

function formatDate(value) {
  const time = toSeconds(value);
  if (time == null) return '—';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date(time * 1000));
}

function formatDateTime(value) {
  const time = toSeconds(value);
  if (time == null) return '—';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(time * 1000));
}

function toDateInputValue(value) {
  const time = toSeconds(value);
  if (time == null) return '';
  const date = new Date(time * 1000);
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, '0'),
    String(date.getDate()).padStart(2, '0'),
  ].join('-');
}

function describeCode(dictionary, code) {
  if (code == null) return 'Unknown code';
  const values = dictionary?.code ?? dictionary;
  if (Array.isArray(values)) {
    const match = values.find((item) => item?.code === code || item?.value === code);
    return match?.description ?? match?.label ?? String(code);
  }
  if (values && typeof values === 'object') {
    const match = values[code];
    return typeof match === 'string'
      ? match
      : match?.description ?? match?.label ?? String(code);
  }
  return String(code);
}

function eventDictionaryKey(eventType) {
  if (eventType === 'imaging_study') return 'imaging_studies';
  if (eventType.endsWith('s')) return eventType;
  return `${eventType}s`;
}

function groupEvents(events) {
  const groups = new Map();
  const rows = Array.isArray(events) ? events : [];

  rows
    .filter((event) => event.event_type === 'encounter')
    .forEach((encounter) => {
      const id = String(encounter.encounter_id ?? encounter.id ?? encounter.start_time);
      groups.set(id, { id, encounter, events: [] });
    });

  rows
    .filter((event) => event.event_type !== 'encounter')
    .forEach((event) => {
      const id = String(event.encounter_id ?? 'unassigned');
      if (!groups.has(id)) {
        groups.set(id, { id, encounter: null, events: [] });
      }
      groups.get(id).events.push(event);
    });

  return [...groups.values()].sort((a, b) => {
    const aTime = toSeconds(a.encounter?.start_time) ?? toSeconds(a.events[0]?.start_time) ?? 0;
    const bTime = toSeconds(b.encounter?.start_time) ?? toSeconds(b.events[0]?.start_time) ?? 0;
    return aTime - bTime;
  });
}

function createTimeScale(minTime, maxTime, encounters) {
  const encounterTimes = encounters.flatMap(({ encounter, events }) => {
    const time = toSeconds(encounter?.start_time ?? events[0]?.start_time);
    return time == null ? [] : [time];
  });
  const anchors = [...new Set([minTime, ...encounterTimes, maxTime])].sort((a, b) => a - b);
  const weights = anchors.slice(1).map((time, index) => {
    const span = (time - anchors[index]) / YEAR;
    return Math.max(0.45, Math.min(2.5, span));
  });
  const total = weights.reduce((sum, weight) => sum + weight, 0) || 1;
  const offsets = weights.reduce((result, weight) => {
    result.push(result.at(-1) + weight);
    return result;
  }, [0]);

  function coordinate(value) {
    const t = toSeconds(value);
    if (t == null) return 0;
    const bounded = Math.max(anchors[0], Math.min(anchors.at(-1), t));
    const nextAnchor = anchors.findIndex((anchor) => anchor > bounded);
    if (nextAnchor === -1) return total;
    const index = Math.max(0, nextAnchor - 1);
    const span = anchors[index + 1] - anchors[index];
    const fraction = span > 0 ? (bounded - anchors[index]) / span : 0;
    return offsets[index] + fraction * (offsets[index + 1] - offsets[index]);
  }

  function valueAt(coord) {
    const bounded = Math.max(0, Math.min(total, coord));
    const nextOffset = offsets.findIndex((offset) => offset > bounded);
    if (nextOffset === -1) return anchors.at(-1);
    const index = Math.max(0, nextOffset - 1);
    const span = offsets[index + 1] - offsets[index];
    const fraction = span > 0 ? (bounded - offsets[index]) / span : 0;
    return anchors[index] + fraction * (anchors[index + 1] - anchors[index]);
  }

  return { coordinate, valueAt, total, minTime, maxTime };
}

function getEventCounts(childEvents) {
  const counts = {};
  for (const ev of childEvents) {
    const type = ev.event_type || 'other';
    counts[type] = (counts[type] || 0) + 1;
  }
  return counts;
}

export default function Timeline() {
  const { patient, events, schema, cutoff: storeCutoff } = usePatientStore();
  const [selectedId, setSelectedId] = useState(null);
  const [filterCategory, setFilterCategory] = useState('all');
  const [now] = useState(() => Math.floor(Date.now() / 1000));
  const [inspectorHeight, setInspectorHeight] = useState(260);
  const [isMaximized, setIsMaximized] = useState(false);
  const trackRef = useRef(null);

  const rows = useMemo(() => (Array.isArray(events) ? events : []), [events]);
  const encounters = useMemo(() => groupEvents(rows), [rows]);

  const eventTimes = useMemo(() => (
    rows
      .flatMap((e) => [toSeconds(e.start_time), toSeconds(e.stop_time)])
      .filter((t) => t != null)
      .sort((a, b) => a - b)
  ), [rows]);

  const { minTime, maxTime, defaultCutoff } = useMemo(() => {
    const birth = toSeconds(patient?.patient_start_time);
    const firstEv = eventTimes[0];
    const earliest = birth ?? firstEv ?? now - YEAR;
    const lastEv = eventTimes.at(-1);
    const latest = Math.max(earliest + DAY, lastEv ?? now);
    const initialCutoff = Math.min(latest, now);
    return { minTime: earliest, maxTime: latest, defaultCutoff: initialCutoff };
  }, [patient?.patient_start_time, eventTimes, now]);

  const cutoff = storeCutoff ?? defaultCutoff;
  const setCutoff = useCallback((newCutoff) => {
    patientStore.setCutoff(typeof newCutoff === 'function' ? newCutoff(cutoff) : newCutoff);
  }, [cutoff]);

  useEffect(() => {
    if (storeCutoff == null && defaultCutoff != null) {
      patientStore.setCutoff(defaultCutoff);
    }
  }, [storeCutoff, defaultCutoff]);

  useEffect(() => {
    setSelectedId(null);
    setFilterCategory('all');
  }, [patient?.patient_id]);

  const timeScale = useMemo(
    () => createTimeScale(minTime, maxTime, encounters),
    [minTime, maxTime, encounters],
  );

  const trackWidth = Math.max(1200, Math.round(timeScale.total * 220));
  const contentWidth = trackWidth - TRACK_PADDING_X * 2;

  // Start each patient at the newest chronological events. This also runs after
  // a reload when the patient data has finished loading and the track is sized.
  useLayoutEffect(() => {
    const track = trackRef.current;
    if (!track) return undefined;

    const frame = requestAnimationFrame(() => {
      track.scrollLeft = track.scrollWidth - track.clientWidth;
    });

    return () => cancelAnimationFrame(frame);
  }, [patient?.patient_id, encounters.length, minTime, maxTime, trackWidth]);

  const timeToX = useCallback((value) => {
    const coord = timeScale.coordinate(value);
    return TRACK_PADDING_X + (coord / timeScale.total) * contentWidth;
  }, [timeScale, contentWidth]);

  const xToTime = useCallback((xPixel) => {
    const clampedX = Math.max(TRACK_PADDING_X, Math.min(TRACK_PADDING_X + contentWidth, xPixel));
    const coord = ((clampedX - TRACK_PADDING_X) / contentWidth) * timeScale.total;
    return Math.round(timeScale.valueAt(coord));
  }, [timeScale, contentWidth]);

  // Compute collision-free lanes for encounters
  const { positionedEncounters, numLanes } = useMemo(() => {
    const laneEnds = [];
    const items = encounters.map((item) => {
      const time = toSeconds(item.encounter?.start_time ?? item.events[0]?.start_time) ?? minTime;
      const x = timeToX(time);
      let lane = 0;
      while (lane < laneEnds.length && laneEnds[lane] > x - 12) {
        lane++;
      }
      laneEnds[lane] = x + CARD_WIDTH;
      return { ...item, x, lane, time };
    });
    return { positionedEncounters: items, numLanes: Math.max(1, laneEnds.length) };
  }, [encounters, timeToX, minTime]);

  const cutoffX = timeToX(cutoff);

  // Year ticks on the ruler
  const rulerTicks = useMemo(() => {
    const startYear = new Date(minTime * 1000).getFullYear();
    const endYear = new Date(maxTime * 1000).getFullYear();
    const span = endYear - startYear;
    const step = span > 40 ? 10 : span > 20 ? 5 : span > 8 ? 2 : 1;
    const ticks = [];
    for (let y = Math.ceil(startYear / step) * step; y <= endYear; y += step) {
      const time = Date.UTC(y, 0, 1) / 1000;
      if (time >= minTime && time <= maxTime) {
        ticks.push({ year: y, x: timeToX(time) });
      }
    }
    return ticks;
  }, [minTime, maxTime, timeToX]);

  // Global event counts for legend
  const globalEventCounts = useMemo(() => {
    const counts = {};
    for (const ev of rows) {
      const type = ev.event_type || 'other';
      counts[type] = (counts[type] || 0) + 1;
    }
    return counts;
  }, [rows]);

  // Active selected encounter details
  const selectedGroup = useMemo(
    () => encounters.find((e) => e.id === selectedId) ?? null,
    [encounters, selectedId],
  );

  const selectedChildEvents = useMemo(() => {
    if (!selectedGroup) return [];
    if (filterCategory === 'all') return selectedGroup.events;
    return selectedGroup.events.filter((e) => e.event_type === filterCategory);
  }, [selectedGroup, filterCategory]);

  // Scrubbing on ruler
  const handleRulerMouseDown = (e) => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect) return;
    const update = (clientX) => {
      const scrollLeft = trackRef.current?.scrollLeft ?? 0;
      const x = clientX - rect.left + scrollLeft;
      setCutoff(xToTime(x));
    };
    update(e.clientX);

    const onMouseMove = (moveEvent) => update(moveEvent.clientX);
    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  // Drag-to-resize inspector height
  const handleResizerMouseDown = (e) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = inspectorHeight;

    const onMouseMove = (moveEvent) => {
      const deltaY = startY - moveEvent.clientY;
      const nextHeight = Math.max(130, Math.min(window.innerHeight * 0.75, startHeight + deltaY));
      setInspectorHeight(nextHeight);
      setIsMaximized(false);
    };

    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  const toggleMaximize = () => {
    if (isMaximized) {
      setInspectorHeight(260);
      setIsMaximized(false);
    } else {
      setInspectorHeight(440);
      setIsMaximized(true);
    }
  };

  if (!patient) {
    return <div className="timeline-empty">Waiting for a patient…</div>;
  }

  const selectedItem = selectedGroup?.encounter ?? selectedGroup?.events[0];
  const selectedIsFuture = (toSeconds(selectedItem?.start_time) ?? 0) > cutoff;

  return (
    <section className="timeline-panel" aria-label="Patient event timeline">
      {/* Top Header */}
      <header className="tl-header">
        <div className="tl-header-left">
          <span className="tl-kicker">PATIENT EVENT TIMELINE</span>
          <div className="tl-header-title-row">
            <h2>{encounters.length} Encounters</h2>
            <span className="tl-dot-sep">·</span>
            <span className="tl-subcount">{rows.length} Total Events</span>
            <span className="tl-time-range">
              ({formatDate(minTime)} – {formatDate(maxTime)})
            </span>
          </div>
        </div>

        {/* Legend Filter Pills */}
        <div className="tl-legend">
          {Object.entries(COLORS).map(([type, color]) => {
            const count = type === 'encounter' ? encounters.length : globalEventCounts[type] || 0;
            if (count === 0 && type !== 'encounter') return null;
            const isActive = filterCategory === type;
            return (
              <button
                key={type}
                type="button"
                className={`tl-legend-pill ${isActive ? 'is-active' : ''}`}
                style={{ '--pill-color': color }}
                onClick={() => setFilterCategory(isActive ? 'all' : type)}
                title={`Filter by ${EVENT_LABELS[type] || type} (${count})`}
              >
                <span className="tl-legend-dot" />
                <span>{EVENT_LABELS[type] || type}</span>
                <span className="tl-legend-count">{count}</span>
              </button>
            );
          })}
        </div>

        {/* Cutoff Controls */}
        <div className="tl-cutoff-controls">
          <div className="tl-cutoff-input-group">
            <span className="tl-cutoff-label">Cut-off date:</span>
            <input
              type="date"
              className="tl-date-input"
              value={toDateInputValue(cutoff)}
              onChange={(e) => {
                if (!e.target.value) return;
                setCutoff(Date.parse(`${e.target.value}T00:00:00Z`) / 1000);
              }}
              title="Filter future clinical events beyond this date"
            />
          </div>
          <button
            type="button"
            className="tl-btn-reset"
            onClick={() => setCutoff(maxTime)}
            title="Set cutoff to latest event"
          >
            All Time
          </button>
        </div>
      </header>

      {/* Horizontal Timeline Track */}
      <div className="tl-track-container" ref={trackRef}>
        <div
          className="tl-canvas"
          style={{
            width: `${trackWidth}px`,
            minHeight: `${Math.max(220, 110 + numLanes * LANE_HEIGHT)}px`,
          }}
        >
          {/* Time Ruler */}
          <div
            className="tl-ruler"
            onMouseDown={handleRulerMouseDown}
            title="Click or drag to change cutoff date"
          >
            <div className="tl-ruler-rail" />

            {/* Birth / Start Marker */}
            <div className="tl-ruler-special-tick" style={{ left: `${TRACK_PADDING_X}px` }}>
              <div className="tl-tick-line" />
              <span className="tl-tick-label">Birth · {formatDate(patient?.patient_start_time)}</span>
            </div>

            {/* Year Ticks */}
            {rulerTicks.map(({ year, x }) => (
              <div key={year} className="tl-ruler-tick" style={{ left: `${x}px` }}>
                <div className="tl-tick-line" />
                <span className="tl-tick-label">{year}</span>
              </div>
            ))}

            {/* Cutoff Flag on Ruler */}
            <div className="tl-cutoff-flag" style={{ left: `${cutoffX}px` }}>
              <span className="tl-cutoff-badge">Cut-off: {formatDate(cutoff)}</span>
              <div className="tl-cutoff-handle" />
            </div>
          </div>

          {/* Full-Height Vertical Cutoff Line */}
          <div className="tl-cutoff-line" style={{ left: `${cutoffX}px` }} />

          {/* Shaded Future Area */}
          <div
            className="tl-future-shading"
            style={{
              left: `${cutoffX}px`,
              width: `${Math.max(0, trackWidth - cutoffX)}px`,
            }}
          >
            <span className="tl-future-watermark">Future (Filtered)</span>
          </div>

          {/* Encounter Cards Lane */}
          <div
            className="tl-encounters-lane"
            style={{ height: `${numLanes * LANE_HEIGHT + 24}px` }}
          >
            {positionedEncounters.map((item) => {
              const enc = item.encounter;
              const isSelected = item.id === selectedId;
              const isFuture = item.time > cutoff;
              const counts = getEventCounts(item.events);
              const title = enc
                ? describeCode(schema?.encounters, enc.code)
                : 'Outpatient / External Events';

              return (
                <div
                  key={item.id}
                  className={`tl-encounter-card ${isSelected ? 'is-selected' : ''} ${isFuture ? 'is-future' : ''}`}
                  style={{
                    left: `${item.x}px`,
                    top: `${item.lane * LANE_HEIGHT + 10}px`,
                    width: `${CARD_WIDTH}px`,
                  }}
                  onClick={() => setSelectedId(item.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setSelectedId(item.id);
                    }
                  }}
                >
                  <div className="tl-enc-header">
                    <span className="tl-enc-date">{formatDate(item.time)}</span>
                    {isFuture && <span className="tl-future-tag">FUTURE</span>}
                  </div>

                  <strong className="tl-enc-title" title={title}>
                    {title}
                  </strong>

                  {enc?.code && (
                    <span className="tl-enc-code">Code: {enc.code}</span>
                  )}

                  {/* Child event summary badges */}
                  <div className="tl-enc-chips">
                    {Object.entries(counts).map(([type, cnt]) => (
                      <span
                        key={type}
                        className="tl-chip"
                        style={{ '--chip-color': COLORS[type] || '#71c7aa' }}
                        title={`${cnt} ${EVENT_LABELS[type] || type}`}
                      >
                        {cnt} {type.slice(0, 4)}
                      </span>
                    ))}
                    {item.events.length === 0 && (
                      <span className="tl-chip tl-chip-muted">0 events</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Selected Encounter Inspector / Event Explorer */}
      <footer
        className={`tl-inspector ${isMaximized ? 'is-maximized' : ''}`}
        style={selectedGroup ? { height: `${inspectorHeight}px` } : undefined}
      >
        {selectedGroup && (
          <div
            className="tl-inspector-resizer"
            onMouseDown={handleResizerMouseDown}
            onDoubleClick={toggleMaximize}
            title="Drag to resize inspector height (Double click to toggle expand)"
          >
            <div className="tl-resizer-grip" />
          </div>
        )}
        {selectedGroup ? (
          <div className="tl-inspector-content">
            {/* Inspector Header */}
            <div className="tl-inspector-header">
              <div className="tl-inspector-meta">
                <span className="tl-inspector-tag">
                  {selectedGroup.encounter ? 'ENCOUNTER DETAILS' : 'INDEPENDENT EVENTS'}
                </span>
                <h3 className="tl-inspector-title">
                  {selectedGroup.encounter
                    ? describeCode(schema?.encounters, selectedGroup.encounter.code)
                    : 'Outpatient / Independent Events'}
                </h3>
                <div className="tl-inspector-sub">
                  <span>📅 {formatDateTime(selectedItem?.start_time)}</span>
                  {selectedGroup.encounter?.stop_time && (
                    <>
                      <span className="tl-dot-sep">·</span>
                      <span>To: {formatDateTime(selectedGroup.encounter.stop_time)}</span>
                    </>
                  )}
                  {selectedItem?.code && (
                    <>
                      <span className="tl-dot-sep">·</span>
                      <code>Code: {selectedItem.code}</code>
                    </>
                  )}
                  {selectedIsFuture && (
                    <span className="tl-inspector-future-badge">Occurs in Future</span>
                  )}
                </div>
              </div>

              {/* Subcategory Filter Tabs inside Inspector */}
              <div className="tl-inspector-actions">
                <div className="tl-category-tabs">
                  <button
                    type="button"
                    className={`tl-tab ${filterCategory === 'all' ? 'is-active' : ''}`}
                    onClick={() => setFilterCategory('all')}
                  >
                    All ({selectedGroup.events.length})
                  </button>
                  {Object.keys(getEventCounts(selectedGroup.events)).map((type) => {
                    const count = selectedGroup.events.filter((e) => e.event_type === type).length;
                    return (
                      <button
                        key={type}
                        type="button"
                        className={`tl-tab ${filterCategory === type ? 'is-active' : ''}`}
                        style={{ '--tab-color': COLORS[type] }}
                        onClick={() => setFilterCategory(type)}
                      >
                        {EVENT_LABELS[type] || type} ({count})
                      </button>
                    );
                  })}
                </div>
                <button
                  type="button"
                  className="tl-btn-icon"
                  onClick={toggleMaximize}
                  title={isMaximized ? 'Restore inspector height' : 'Expand inspector height'}
                >
                  {isMaximized ? '⤡' : '⤢'}
                </button>
                <button
                  type="button"
                  className="tl-btn-icon"
                  onClick={() => setSelectedId(null)}
                  title="Close encounter details"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Event List / Cards */}
            <div className="tl-events-container">
              {selectedChildEvents.length === 0 ? (
                <div className="tl-no-events">
                  {selectedGroup.events.length === 0
                    ? 'No secondary events (conditions, medications, observations) recorded for this encounter.'
                    : `No ${EVENT_LABELS[filterCategory] || filterCategory} records for this encounter.`}
                </div>
              ) : (
                <div className="tl-events-grid">
                  {selectedChildEvents.map((ev, idx) => {
                    const code = ev.code ?? ev.bodysite_code;
                    const dictKey = eventDictionaryKey(ev.event_type);
                    const desc = describeCode(schema?.[dictKey], code);
                    const color = COLORS[ev.event_type] || '#71c7aa';
                    const evFuture = (toSeconds(ev.start_time) ?? 0) > cutoff;

                    return (
                      <div
                        key={`${ev.event_type}-${ev.index ?? idx}`}
                        className={`tl-event-card ${evFuture ? 'is-future' : ''}`}
                        style={{ '--card-color': color }}
                      >
                        <div className="tl-event-card-top">
                          <span className="tl-event-badge" style={{ backgroundColor: color }}>
                            {EVENT_LABELS[ev.event_type] || ev.event_type}
                          </span>
                          <span className="tl-event-date">{formatDate(ev.start_time)}</span>
                          {evFuture && <span className="tl-event-future-tag">FUTURE</span>}
                        </div>

                        <strong className="tl-event-title" title={desc}>
                          {desc}
                        </strong>

                        <div className="tl-event-footer">
                          <span className="tl-event-code">Code: {code ?? '—'}</span>
                          {ev.value != null && (
                            <span className="tl-event-value">
                              Value: <strong>{ev.value} {ev.units ?? ''}</strong>
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="tl-inspector-prompt">
            <span>💡 Click any encounter above to inspect associated conditions, medications, procedures, and vitals.</span>
          </div>
        )}
      </footer>
    </section>
  );
}
