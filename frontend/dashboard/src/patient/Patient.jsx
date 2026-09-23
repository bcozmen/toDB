import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import '@shoelace-style/shoelace/dist/components/avatar/avatar.js';
import '@shoelace-style/shoelace/dist/components/button/button.js';
import { api } from '../api';
import { patientStore, usePatientStore } from './patientStore';
import './patient.css';

const TODAY = 1790812800;

function toUnixSeconds(value) {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed / 1000;
}

function formatDate(value) {
  const seconds = toUnixSeconds(value);
  if (seconds === null) return 'Unknown';
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(seconds * 1000));
}

function getAge(patient) {
  const birth = toUnixSeconds(patient?.patient_start_time);
  if (birth === null) return 'Unknown';
  const end = toUnixSeconds(patient.patient_stop_time) ?? TODAY;
  const age = Math.max(0, Math.floor((end - birth) / (365.2425 * 24 * 60 * 60)));
  return patient.patient_stop_time ? `${age} years · deceased` : `${age} years old`;
}

function PatientValueMenu({ label, values, editable = false, anchor, onSelect, onClose }) {
  const menuRef = useRef(null);
  const [draft, setDraft] = useState('');
  const [position, setPosition] = useState({ left: anchor.left, top: anchor.bottom + 8 });

  useEffect(() => {
    const updatePosition = () => setPosition({ left: anchor.left, top: anchor.bottom + 8 });
    const closeOnOutsideClick = (event) => {
      if (!menuRef.current?.contains(event.target)) onClose();
    };
    const closeOnEscape = (event) => event.key === 'Escape' && onClose();
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    document.addEventListener('mousedown', closeOnOutsideClick);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
      document.removeEventListener('mousedown', closeOnOutsideClick);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [anchor, onClose]);

  return createPortal(
    <div ref={menuRef} className="patient-value-menu" style={position} role="dialog" aria-label={`${label} possible values`}>
      <div className="patient-value-menu-heading">
        <span>{label} values</span>
        <button type="button" onClick={onClose} aria-label="Close values">×</button>
      </div>
      <div className="patient-value-list">
        {editable && (
          <form className="patient-value-form" onSubmit={(event) => { event.preventDefault(); onSelect(draft === '' ? null : Number(draft)); }}>
            <label htmlFor="patient-income-input">Enter income</label>
            <input
              id="patient-income-input"
              type="number"
              min="0"
              step="0.01"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              autoFocus
            />
            <button type="submit">Apply value</button>
          </form>
        )}
        {values.length === 0 && <div className="patient-value-empty">No dictionary values available.</div>}
        {values.map((option) => (
          <button type="button" key={String(option)} onClick={() => onSelect(option)}>{String(option)}</button>
        ))}
      </div>
    </div>,
    document.body,
  );
}

function Detail({ field, label, value, values, editable = false, onOpen }) {
  return (
    <button type="button" className="patient-detail" onClick={(event) => onOpen(field, label, values, editable, event.currentTarget)}>
      <span>{label}</span>
      <strong>{value ?? 'Unknown'}</strong>
    </button>
  );
}

export default function Patient() {
  const { patient, events, loading, error } = usePatientStore();
  const [patientDictionary, setPatientDictionary] = useState({});
  const [openDetail, setOpenDetail] = useState(null);

  const loadPatient = useCallback(async () => {
    patientStore.setLoading();
    try {
      const data = await api.getRandomPatient();
      patientStore.setData(data.patient, data.events, { ...data.patient });
    } catch (loadError) {
      patientStore.setError(loadError);
    }
  }, []);

  useEffect(() => { loadPatient(); }, [loadPatient]);

  useEffect(() => {
    api.getSchema()
      .then((schema) => {
        const dictionaries = Object.fromEntries(schema.tables.map((table) => [table.name.toLowerCase(), table.dictionary ?? {}]));
        patientStore.setSchema(dictionaries);
        setPatientDictionary(dictionaries.patients ?? {});
      })
      .catch(() => setPatientDictionary({}));
  }, []);

  const openValues = (field, label, values, editable, element) => {
    setOpenDetail({ field, label, values: Array.isArray(values) ? values : [], editable, anchor: element.getBoundingClientRect() });
  };

  const selectValue = (value) => {
    patientStore.updatePatientField(openDetail.field, value);
    setOpenDetail(null);
  };

  if (loading && !patient) return <div className="patient-state">Loading patient…</div>;
  if (error && !patient) return <div className="patient-state patient-error">Could not load patient. <button type="button" onClick={loadPatient}>Retry</button></div>;

  return (
    <section className="patient-panel" aria-label="Patient details">
      <header className="patient-summary">
        <sl-avatar className="patient-avatar" label="Patient avatar" />
        <div className="patient-identity">
          <span className="patient-kicker">PATIENT PROFILE</span>
          <h2>{patient?.patient_id ?? 'Unknown patient'}</h2>
          <div className="patient-primary-details">
            <span>Born {formatDate(patient?.patient_start_time)}</span>
            <span>{getAge(patient)}</span>
          </div>
        </div>
        <div className="patient-details-grid">
          <Detail field="gender" label="Gender" value={patient?.gender} values={patientDictionary.gender} onOpen={openValues} />
          <Detail field="race" label="Race" value={patient?.race} values={patientDictionary.race} onOpen={openValues} />
          <Detail field="ethnicity" label="Ethnicity" value={patient?.ethnicity} values={patientDictionary.ethnicity} onOpen={openValues} />
          <Detail field="birthplace" label="Birthplace" value={patient?.birthplace} values={patientDictionary.birthplace} onOpen={openValues} />
          <Detail field="city" label="City" value={patient?.city} values={patientDictionary.city} onOpen={openValues} />
            <Detail field="income" label="Income" value={patient?.income == null ? null : `$${Number(patient.income).toLocaleString()}`} values={[]} editable onOpen={openValues} />
        </div>
          <button className="new-patient-button" type="button" onClick={loadPatient} disabled={loading}>
            <span aria-hidden="true">↻</span>{loading ? 'Loading…' : 'New patient'}
          </button>
      </header>
      {openDetail && <PatientValueMenu {...openDetail} onSelect={selectValue} onClose={() => setOpenDetail(null)} />}
      <div className="patient-timeline-placeholder">
        <span>EVENT TIMELINE</span>
        <p>{patient ? `${Array.isArray(events) ? events.length : 0} events ready` : 'No events loaded'}</p>
      </div>
    </section>
  );
}