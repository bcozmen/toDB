import { useSyncExternalStore } from 'react';

// Golden Layout mounts each panel in its own React root, so React Context
// cannot cross the panel boundary. This store can also be consumed by Out ML.
let snapshot = { patient: null, events: [], schema: {}, groundTruth: {}, loading: true, error: null };
const listeners = new Set();

function emit(nextSnapshot) {
  snapshot = nextSnapshot;
  listeners.forEach((listener) => listener());
}

export const patientStore = {
  getSnapshot: () => snapshot,
  subscribe: (listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  setLoading: () => emit({ ...snapshot, loading: true, error: null }),
  setData: (patient, events = [], groundTruth = {}) => emit({ ...snapshot, patient, events, groundTruth, loading: false, error: null }),
  setSchema: (schema) => emit({ ...snapshot, schema }),
  updatePatientField: (field, value) => emit({
    ...snapshot,
    patient: { ...snapshot.patient, [field]: value },
    groundTruth: { ...snapshot.groundTruth, [field]: value },
  }),
  setError: (error) => emit({ ...snapshot, loading: false, error }),
};

export function usePatientStore() {
  return useSyncExternalStore(patientStore.subscribe, patientStore.getSnapshot);
}