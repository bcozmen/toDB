import { useSyncExternalStore } from 'react';
import { api } from '../api';

function toSeconds(value) {
  if (value == null || value === '') return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : Math.floor(parsed / 1000);
}

function computeDefaultCutoff(patient, events) {
  const now = Math.floor(Date.now() / 1000);
  const rows = Array.isArray(events) ? events : [];
  const eventTimes = rows
    .flatMap((e) => [toSeconds(e.start_time), toSeconds(e.stop_time)])
    .filter((t) => t != null)
    .sort((a, b) => a - b);
  const birth = toSeconds(patient?.patient_start_time);
  const earliest = birth ?? eventTimes[0] ?? now - 365 * 86400;
  const lastEv = eventTimes.at(-1);
  const latest = Math.max(earliest + 86400, lastEv ?? now);
  return Math.min(latest, now);
}

let aiDebounceTimer = null;

// Golden Layout mounts each panel in its own React root, so React Context
// cannot cross the panel boundary. This store can also be consumed by Out ML.
let snapshot = {
  patient: null,
  events: [],
  schema: {},
  groundTruth: {},
  loading: true,
  error: null,
  cutoff: null,
  aiInsights: null,
  aiLoading: false,
  aiError: null,
};
const listeners = new Set();

function emit(nextSnapshot) {
  snapshot = nextSnapshot;
  listeners.forEach((listener) => listener());
}

async function fetchInsightsForCurrentState() {
  const s = snapshot;
  if (!s.patient) return;
  const death = toSeconds(s.patient.patient_stop_time);
  const cutoff = s.cutoff ?? Math.floor(Date.now() / 1000);

  // If patient died on or before cutoff date, they are deceased as of cutoff
  if (death !== null && death <= cutoff) {
    emit({ ...snapshot, aiInsights: null, aiLoading: false, aiError: null });
    return;
  }

  emit({ ...snapshot, aiLoading: true, aiError: null });
  try {
    const filteredEvents = (Array.isArray(s.events) ? s.events : []).filter((e) => {
      const t = toSeconds(e.start_time);
      return t == null || t <= cutoff;
    });
    const data = await api.getAiInsights(s.patient, filteredEvents);
    emit({ ...snapshot, aiInsights: data, aiLoading: false, aiError: null });
  } catch (err) {
    emit({
      ...snapshot,
      aiError: err?.response?.data?.detail || err?.message || 'Failed to fetch AI insights',
      aiLoading: false,
    });
  }
}

export const patientStore = {
  getSnapshot: () => snapshot,
  subscribe: (listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  setLoading: () => emit({ ...snapshot, loading: true, error: null }),
  setData: (patient, events = [], groundTruth = {}) => {
    const defaultCutoff = computeDefaultCutoff(patient, events);
    emit({
      ...snapshot,
      patient,
      events,
      groundTruth,
      cutoff: defaultCutoff,
      aiInsights: null,
      aiError: null,
      loading: false,
      error: null,
    });
    fetchInsightsForCurrentState();
  },
  setCutoff: (cutoff) => {
    emit({ ...snapshot, cutoff });
    if (aiDebounceTimer) clearTimeout(aiDebounceTimer);
    aiDebounceTimer = setTimeout(() => {
      fetchInsightsForCurrentState();
    }, 400);
  },
  fetchAiInsights: () => {
    if (aiDebounceTimer) clearTimeout(aiDebounceTimer);
    fetchInsightsForCurrentState();
  },
  setSchema: (schema) => emit({ ...snapshot, schema }),
  updatePatientField: (field, value) => {
    const updatedPatient = { ...snapshot.patient, [field]: value };
    emit({
      ...snapshot,
      patient: updatedPatient,
      groundTruth: { ...snapshot.groundTruth, [field]: value },
    });
    if (aiDebounceTimer) clearTimeout(aiDebounceTimer);
    aiDebounceTimer = setTimeout(() => {
      fetchInsightsForCurrentState();
    }, 400);
  },
  setError: (error) => emit({ ...snapshot, loading: false, error }),
};

export function usePatientStore() {
  return useSyncExternalStore(patientStore.subscribe, patientStore.getSnapshot);
}