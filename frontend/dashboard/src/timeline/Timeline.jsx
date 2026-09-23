import React, { useEffect, useMemo, useState } from 'react';
import '@shoelace-style/shoelace/dist/components/card/card.js';
import { usePatientStore } from '../patient/patientStore';
import './timeline.css';

export const EVENT_TABLES = [
  ['encounters', 'encounter'],
  ['conditions', 'condition'],
  ['medications', 'medication'],
  ['procedures', 'procedure'],
  ['observations', 'observation'],
  ['immunizations', 'immunization'],
  ['allergies', 'allergy'],
  ['careplans', 'careplan'],
  ['imaging_studies', 'imaging_study'],
];

export const COLORS = {
  encounter: '#62c5a2',
  condition: '#ef8f8f',
  medication: '#d9a85f',
  procedure: '#9f9bea',
  observation: '#65b6d9',
  immunization: '#d483bc',
  allergy: '#e57b60',
  careplan: '#b2c96b',
  imaging_study: '#e3c76d',
};

const DAY = 86400;
const YEAR = 365 * DAY;
const TIMELINE_END_PADDING = 240;
const TIMELINE_SLIDER_INSET = 7;
const DEFAULT_EVENT_COLOR = '#8ca0aa';

function toSeconds(value) {
  if (value == null || value === '') return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;

  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed / 1000;
}

function formatDate(value) {
  const time = toSeconds(value);
  return time == null
    ? 'Unknown'
    : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(time * 1000));
}

function toDateInputValue(value) {
  const time = toSeconds(value);
  if (time == null) return '';

  const date = new Date(time * 1000);
  return [
    date.getUTCFullYear(),
    String(date.getUTCMonth() + 1).padStart(2, '0'),
    String(date.getUTCDate()).padStart(2, '0'),
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
  return eventType === 'imaging_study' ? 'imaging_studies' : `${eventType}s`;
}

function groupEvents(events) {
  const groups = new Map();
  const rows = Array.isArray(events) ? events : [];

  rows
    .filter((event) => event.event_type === 'encounter')
    .forEach((encounter) => {
      const id = String(encounter.encounter_id ?? encounter.id ?? encounter.start_time);
      groups.set(id, { encounter, events: [] });
    });

  rows
    .filter((event) => event.event_type !== 'encounter')
    .forEach((event) => {
      const id = String(event.encounter_id ?? 'unassigned');
      if (!groups.has(id)) groups.set(id, { encounter: null, events: [] });
      groups.get(id).events.push(event);
    });

  return [...groups.values()].sort((left, right) => {
    const leftTime = toSeconds(left.encounter?.start_time) ?? toSeconds(left.events[0]?.start_time) ?? 0;
    const rightTime = toSeconds(right.encounter?.start_time) ?? toSeconds(right.events[0]?.start_time) ?? 0;
    return leftTime - rightTime;
  });
}

function createTimeScale(minTime, maxTime, encounters) {
  const encounterTimes = encounters.flatMap(({ encounter, events }) => {
    const time = toSeconds(encounter?.start_time ?? events[0]?.start_time);
    return time == null ? [] : [time];
  });
  const anchors = [...new Set([minTime, ...encounterTimes, maxTime])].sort((a, b) => a - b);
  const weights = anchors.slice(1).map((time, index) => (
    Math.max(0.45, Math.min(2, (time - anchors[index]) / YEAR))
  ));
  const total = weights.reduce((sum, weight) => sum + weight, 0) || 1;
  const offsets = weights.reduce((result, weight) => {
    result.push(result.at(-1) + weight);
    return result;
  }, [0]);

  function coordinate(value) {
    const boundedTime = Math.max(anchors[0], Math.min(anchors.at(-1), toSeconds(value) ?? anchors[0]));
    const nextAnchor = anchors.findIndex((anchor) => anchor > boundedTime);
    const index = Math.max(0, Math.min(anchors.length - 2, nextAnchor - 1));
    const span = anchors[index + 1] - anchors[index];
    const fraction = span > 0 ? (boundedTime - anchors[index]) / span : 0;
    return offsets[index] + fraction * (offsets[index + 1] - offsets[index]);
  }

  function valueAt(value) {
    const boundedCoordinate = Math.max(0, Math.min(total, value));
    const nextOffset = offsets.findIndex((offset) => offset > boundedCoordinate);
    const index = Math.max(0, Math.min(offsets.length - 2, nextOffset - 1));
    const span = offsets[index + 1] - offsets[index];
    const fraction = span > 0 ? (boundedCoordinate - offsets[index]) / span : 0;
    return anchors[index] + fraction * (anchors[index + 1] - anchors[index]);
  }

  return { coordinate, valueAt, total };
}

function getGroupId(encounter, events, index) {
  return String(encounter?.encounter_id ?? events[0]?.encounter_id ?? index);
}

function EventCard({ event, dictionary, top, color, isFuture }) {
  const code = event.code ?? event.bodysite_code;
  const description = describeCode(dictionary, code);

  return (
    <sl-card
      className={`timeline-card event-card ${isFuture ? 'is-future' : ''}`}
      style={{ top: `${top}px`, '--event-color': color }}
    >
      <div className="timeline-card-kind">{event.event_type}</div>
      <strong>{formatDate(event.start_time)}</strong>
      <span>{formatDate(event.stop_time)}</span>
      <span><b>Code:</b> {code ?? 'Unknown'}</span>
      <span className="timeline-description" title={description}>{description}</span>
      <span><b>Value:</b> {event.value ?? '—'} {event.units ?? ''}</span>
    </sl-card>
  );
}

function EncounterCard({ encounter, events, cutoff, expanded, onToggle, position, width, schema }) {
  const item = encounter ?? events[0];
  const type = encounter?.event_type ?? 'encounter';
  const isFuture = (toSeconds(item?.start_time) ?? 0) > cutoff;

  return (
    <sl-card
      className={`encounter-card ${expanded ? 'is-open' : ''} ${isFuture ? 'is-future' : ''}`}
      role="button"
      tabIndex="0"
      onClick={onToggle}
      onKeyDown={(event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        onToggle();
      }}
      style={{
        left: `${position(item?.start_time)}%`,
        width: `${width(item?.start_time, encounter?.stop_time)}%`,
        '--event-color': COLORS[type],
      }}
    >
      <span className="card-type">ENCOUNTER</span>
      <strong>{formatDate(item?.start_time)}</strong>
      <span>{formatDate(encounter?.stop_time)}</span>
      <span>Code: {item?.code ?? 'Unknown'}</span>
      <span className="timeline-description">
        {describeCode(schema?.encounters, item?.code)}
      </span>
    </sl-card>
  );
}

function EventGroup({ encounter, events, index, cutoff, schema, expanded, position }) {
  if (!expanded) return null;

  const groupId = getGroupId(encounter, events, index);
  const start = encounter?.start_time ?? events[0]?.start_time;

  return (
    <div className="event-group" style={{ left: `${position(start)}%` }}>
      {events.map((event, eventIndex) => (
        <EventCard
          key={`${groupId}-${event.event_type}-${event.index ?? eventIndex}`}
          event={event}
          dictionary={schema?.[eventDictionaryKey(event.event_type)]}
          top={eventIndex * 132}
          color={COLORS[event.event_type] ?? DEFAULT_EVENT_COLOR}
          isFuture={(toSeconds(event.start_time) ?? 0) > cutoff}
        />
      ))}
    </div>
  );
}

function TimelineHeader({ encounterCount, eventCount, cutoff, onCutoffChange }) {
  return (
    <header className="timeline-header">
      <div>
        <span className="timeline-kicker">EVENT TIMELINE</span>
        <h2>{encounterCount} encounters · {eventCount} events</h2>
      </div>
      <label>
        Cut-off date
        <input
          type="date"
          value={toDateInputValue(cutoff)}
          onChange={(event) => {
            if (!event.target.value) return;
            onCutoffChange(Date.parse(`${event.target.value}T00:00:00Z`) / 1000);
          }}
        />
      </label>
    </header>
  );
}

function TimelineRuler({ minTime, maxTime, cutoff, scale, sliderValue, linePosition, lineHeight, onCutoffChange }) {
  return (
    <div className="timeline-ruler" aria-label="Timeline cutoff control">
      <div className="timeline-ruler-labels">
        <span>{formatDate(minTime)}</span>
        <span>Birth</span>
        <span>{formatDate(maxTime)}</span>
      </div>
      <input
        className="timeline-cutoff-slider"
        type="range"
        min="0"
        max={scale.total}
        step="0.01"
        value={sliderValue}
        onChange={(event) => onCutoffChange(scale.valueAt(Number(event.target.value)))}
        aria-label="Timeline cut-off date"
      />
      <div className="timeline-cutoff-line" style={{ left: `${linePosition}px`, height: `${lineHeight}px` }}>
        <span>{formatDate(cutoff)}</span>
      </div>
    </div>
  );
}

export default function Timeline() {
  const { patient, events, schema } = usePatientStore();
  const [expanded, setExpanded] = useState(() => new Set());
  const [now] = useState(() => Math.floor(Date.now() / 1000));
  const [cutoff, setCutoff] = useState(now);
  const rows = useMemo(() => (Array.isArray(events) ? events : []), [events]);

  const eventTimes = useMemo(() => rows
    .flatMap((event) => [toSeconds(event.start_time), toSeconds(event.stop_time)])
    .filter((time) => time != null)
    .sort((a, b) => a - b), [rows]);
  const lastEvent = eventTimes.at(-1) ?? now;
  const defaultCutoff = Math.min(lastEvent, now);

  useEffect(() => {
    setCutoff(defaultCutoff);
    setExpanded(new Set());
  }, [patient?.patient_id, defaultCutoff]);

  const encounters = useMemo(() => groupEvents(rows), [rows]);
  const { minTime, maxTime } = useMemo(() => {
    const firstEvent = eventTimes[0] ?? toSeconds(patient?.patient_start_time) ?? 0;
    const lastEventTime = eventTimes.at(-1)
      ?? toSeconds(patient?.patient_stop_time)
      ?? now;
    return { minTime: firstEvent, maxTime: Math.max(lastEventTime, firstEvent + DAY) };
  }, [eventTimes, now, patient]);
  const timeScale = useMemo(
    () => createTimeScale(minTime, maxTime, encounters),
    [minTime, maxTime, encounters],
  );

  const timelineWidth = Math.max(1400, timeScale.total * 300);
  const position = (value) => (
    timeScale.coordinate(value) / timeScale.total
    * timelineWidth / (timelineWidth + TIMELINE_END_PADDING) * 100
  );
  const linePosition = (value) => (
    TIMELINE_SLIDER_INSET
    + timeScale.coordinate(value) / timeScale.total
    * (timelineWidth - TIMELINE_SLIDER_INSET * 2)
  );
  const widthBetween = (start, stop) => Math.max(3, position(stop ?? start) - position(start));
  const toggle = (id) => setExpanded((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });
  const expandedHeight = Math.max(0, ...encounters.map(({ encounter, events: childEvents }, index) => (
    expanded.has(getGroupId(encounter, childEvents, index))
      ? 28 + childEvents.length * 132
      : 0
  )));

  if (!patient) return <div className="timeline-empty">Waiting for a patient…</div>;

  return (
    <section className="timeline-panel" aria-label="Patient event timeline">
      <TimelineHeader
        encounterCount={encounters.length}
        eventCount={rows.length}
        cutoff={cutoff}
        onCutoffChange={setCutoff}
      />
      <div className={`timeline-scroll ${expandedHeight > 0 ? 'has-expanded-events' : ''}`}>
        <div
          className="timeline-canvas"
          style={{
            '--timeline-end-padding': `${TIMELINE_END_PADDING}px`,
            width: `${timelineWidth + TIMELINE_END_PADDING}px`,
            height: `${Math.max(300, 238 + expandedHeight)}px`,
          }}
        >
          <TimelineRuler
            minTime={minTime}
            maxTime={maxTime}
            cutoff={cutoff}
            scale={timeScale}
            sliderValue={timeScale.coordinate(cutoff)}
            linePosition={linePosition(cutoff)}
            lineHeight={expandedHeight + 220}
            onCutoffChange={setCutoff}
          />
          <div className="timeline-encounter-row">
            {encounters.map(({ encounter, events: childEvents }, index) => {
              const id = getGroupId(encounter, childEvents, index);
              return (
                <EncounterCard
                  key={id}
                  encounter={encounter}
                  events={childEvents}
                  cutoff={cutoff}
                  expanded={expanded.has(id)}
                  onToggle={() => toggle(id)}
                  position={position}
                  width={widthBetween}
                  schema={schema}
                />
              );
            })}
          </div>
          {expandedHeight > 0 && (
            <div className="timeline-event-row" style={{ minHeight: `${expandedHeight}px` }}>
              {encounters.map(({ encounter, events: childEvents }, index) => (
                <EventGroup
                  key={getGroupId(encounter, childEvents, index)}
                  encounter={encounter}
                  events={childEvents}
                  index={index}
                  cutoff={cutoff}
                  schema={schema}
                  expanded={expanded.has(getGroupId(encounter, childEvents, index))}
                  position={position}
                />
              ))}
            </div>
          )}
          <div className="timeline-axis">
            <span>{formatDate(minTime)}</span>
            <span>Birth · {formatDate(patient.patient_start_time)}</span>
            <span>{formatDate(maxTime)}</span>
          </div>
        </div>
      </div>
    </section>
  );
}
