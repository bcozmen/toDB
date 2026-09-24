import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { api } from '../api';
import { patientStore, usePatientStore } from './patientStore';
import './patient.css';

const TODAY = 1790812800; // Simulated reference date (~2026)

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

function getPatientMeta(patient) {
  const birth = toUnixSeconds(patient?.patient_start_time);
  const end = toUnixSeconds(patient?.patient_stop_time);
  const isDeceased = end !== null;
  if (birth === null) {
    return { ageText: 'Unknown age', isDeceased };
  }
  const refTime = end ?? TODAY;
  const age = Math.max(0, Math.floor((refTime - birth) / (365.2425 * 24 * 60 * 60)));
  return {
    ageText: isDeceased ? `${age} yrs (deceased)` : `${age} yrs old`,
    isDeceased,
  };
}

function formatDisplayValue(field, value) {
  if (value === null || value === undefined || value === '') return null;
  if (field === 'income') {
    const num = Number(value);
    return Number.isFinite(num) ? `$${num.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : String(value);
  }
  if (field === 'gender') {
    const upper = String(value).toUpperCase();
    if (upper === 'M') return 'Male';
    if (upper === 'F') return 'Female';
    return String(value);
  }
  if (field === 'ethnicity') {
    const lower = String(value).toLowerCase();
    if (lower === 'nonhispanic') return 'Non-Hispanic';
    if (lower === 'hispanic') return 'Hispanic';
    return String(value);
  }
  if (field === 'race') {
    const str = String(value);
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
  return String(value);
}

function PatientAvatar() {
  return (
    <div className="patient-avatar-wrap" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    </div>
  );
}

function PatientValueMenu({ label, field, currentValue, values, editable = false, anchor, onSelect, onClose }) {
  const menuRef = useRef(null);
  const [search, setSearch] = useState('');
  const [incomeDraft, setIncomeDraft] = useState(
    currentValue != null && Number.isFinite(Number(currentValue)) ? String(Number(currentValue)) : ''
  );
  const [position, setPosition] = useState({
    left: Math.max(10, Math.min(window.innerWidth - 290, anchor.left)),
    top: anchor.bottom + 6,
  });

  const showSearch = !editable && values.length > 5;

  const filteredValues = useMemo(() => {
    if (!showSearch || !search.trim()) return values;
    const q = search.toLowerCase().trim();
    return values.filter((val) => String(val).toLowerCase().includes(q));
  }, [values, search, showSearch]);

  useEffect(() => {
    const updatePosition = () => {
      const menuWidth = Math.min(280, window.innerWidth - 20);
      const menuHeight = menuRef.current?.offsetHeight || 250;
      const fitsBelow = anchor.bottom + menuHeight + 12 <= window.innerHeight;
      const top = fitsBelow ? anchor.bottom + 6 : Math.max(10, anchor.top - menuHeight - 6);
      const left = Math.max(10, Math.min(window.innerWidth - menuWidth - 10, anchor.left));
      setPosition({ left, top });
    };

    updatePosition();
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
    };

    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    document.addEventListener('keydown', handleKey);

    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
      document.removeEventListener('keydown', handleKey);
    };
  }, [anchor, onClose]);

  return createPortal(
    <>
      <div className="patient-menu-backdrop" onClick={onClose} aria-hidden="true" />
      <div
        ref={menuRef}
        className="patient-value-menu"
        style={position}
        role="dialog"
        aria-label={`${label} selection menu`}
      >
        <div className="patient-menu-header">
          <div className="patient-menu-title">
            <span>{label}</span>
            {!editable && values.length > 0 && (
              <span className="patient-menu-count">({values.length})</span>
            )}
          </div>
          <button type="button" className="patient-menu-close" onClick={onClose} aria-label="Close menu">
            ×
          </button>
        </div>

        {editable ? (
          <form
            className="patient-income-form"
            onSubmit={(e) => {
              e.preventDefault();
              const val = incomeDraft.trim() === '' ? null : Number(incomeDraft);
              onSelect(Number.isNaN(val) ? null : val);
            }}
          >
            <label className="patient-income-label" htmlFor="patient-income-field">
              Annual Income (USD)
            </label>
            <div className="patient-income-input-wrap">
              <span className="patient-currency-prefix">$</span>
              <input
                id="patient-income-field"
                type="number"
                min="0"
                step="100"
                placeholder="e.g. 55000"
                className="patient-income-input"
                value={incomeDraft}
                onChange={(e) => setIncomeDraft(e.target.value)}
                autoFocus
              />
            </div>
            <div className="patient-income-actions">
              <button
                type="button"
                className="patient-btn-ghost"
                onClick={() => onSelect(null)}
              >
                Clear
              </button>
              <button type="submit" className="patient-btn-save">
                Apply
              </button>
            </div>
          </form>
        ) : (
          <>
            {showSearch && (
              <div className="patient-menu-search">
                <input
                  type="text"
                  className="patient-search-input"
                  placeholder={`Search ${values.length} ${label.toLowerCase()}s…`}
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  autoFocus
                />
              </div>
            )}
            <div className="patient-value-list">
              {filteredValues.length === 0 ? (
                <div className="patient-value-empty">
                  {search ? 'No matching values.' : 'No dictionary values.'}
                </div>
              ) : (
                filteredValues.map((opt) => {
                  const isSelected = String(opt) === String(currentValue);
                  const displayOpt = formatDisplayValue(field, opt);
                  return (
                    <button
                      type="button"
                      key={String(opt)}
                      className={`patient-menu-item ${isSelected ? 'is-selected' : ''}`}
                      onClick={() => onSelect(opt)}
                    >
                      <span>{displayOpt}</span>
                      {isSelected && <span className="patient-check-icon">✓</span>}
                    </button>
                  );
                })
              )}
            </div>
          </>
        )}
      </div>
    </>,
    document.body
  );
}

function DetailCard({ field, label, rawValue, values, editable = false, onOpen }) {
  const formatted = formatDisplayValue(field, rawValue);

  return (
    <button
      type="button"
      className="patient-detail-card"
      onClick={(e) => onOpen(field, label, rawValue, values, editable, e.currentTarget)}
      title={formatted ? `${label}: ${formatted} (click to change)` : `Set ${label}`}
      aria-label={`Edit ${label}`}
    >
      <div className="patient-card-header">
        <span className="patient-card-label">{label}</span>
        <span className="patient-card-chevron" aria-hidden="true">▾</span>
      </div>
      <strong className={`patient-card-value ${formatted ? '' : 'is-empty'}`}>
        {formatted ?? '—'}
      </strong>
    </button>
  );
}

export default function Patient() {
  const { patient, events, loading, error } = usePatientStore();
  const [patientDictionary, setPatientDictionary] = useState({});
  const [openDetail, setOpenDetail] = useState(null);
  const [copied, setCopied] = useState(false);

  const loadPatient = useCallback(async () => {
    patientStore.setLoading();
    try {
      const data = await api.getRandomPatient();
      patientStore.setData(data.patient, data.events, { ...data.patient });
    } catch (loadError) {
      patientStore.setError(loadError);
    }
  }, []);

  useEffect(() => {
    loadPatient();
  }, [loadPatient]);

  useEffect(() => {
    api.getSchema()
      .then((schema) => {
        const dictionaries = Object.fromEntries(
          schema.tables.map((table) => [table.name.toLowerCase(), table.dictionary ?? {}])
        );
        patientStore.setSchema(dictionaries);
        setPatientDictionary(dictionaries.patients ?? {});
      })
      .catch(() => setPatientDictionary({}));
  }, []);

  const openValues = (field, label, rawValue, values, editable, element) => {
    setOpenDetail({
      field,
      label,
      currentValue: rawValue,
      values: Array.isArray(values) ? values : [],
      editable,
      anchor: element.getBoundingClientRect(),
    });
  };

  const selectValue = (value) => {
    if (openDetail) {
      patientStore.updatePatientField(openDetail.field, value);
      setOpenDetail(null);
    }
  };

  const copyPatientId = () => {
    if (!patient?.patient_id) return;
    navigator.clipboard.writeText(String(patient.patient_id));
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  if (loading && !patient) {
    return (
      <div className="patient-state">
        <div className="patient-spinner" />
        <span>Loading patient profile…</span>
      </div>
    );
  }

  if (error && !patient) {
    return (
      <div className="patient-state">
        <div className="patient-error-box">
          <span className="patient-error-msg">Could not load patient record.</span>
          <button type="button" className="patient-retry-btn" onClick={loadPatient}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  const { ageText, isDeceased } = getPatientMeta(patient);

  return (
    <section className="patient-panel" aria-label="Patient details">
      <header className="patient-header">
        <div className="patient-profile-main">
          <PatientAvatar />

          <div className="patient-identity">
            <div className="patient-meta-top">
              <span className="patient-kicker">PATIENT PROFILE</span>
              <span className={`patient-status-pill ${isDeceased ? 'status-deceased' : 'status-active'}`}>
                {isDeceased ? 'Deceased' : 'Active'}
              </span>
            </div>

            <div className="patient-title-row">
              <h2 className="patient-id-heading" title={String(patient?.patient_id ?? 'Unknown patient')}>
                {patient?.patient_id ?? 'Unknown'}
              </h2>
              {patient?.patient_id && (
                <button
                  type="button"
                  className="patient-copy-btn"
                  onClick={copyPatientId}
                  title="Copy patient ID"
                >
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              )}
            </div>

            <div className="patient-primary-details">
              <span className="patient-detail-tag">Born {formatDate(patient?.patient_start_time)}</span>
              <span className="patient-detail-tag">{ageText}</span>
              {patient?.n_events != null && (
                <span className="patient-detail-tag">{patient.n_events} events</span>
              )}
            </div>
          </div>
        </div>

        <button
          className={`new-patient-button ${loading ? 'is-loading' : ''}`}
          type="button"
          onClick={loadPatient}
          disabled={loading}
          title="Fetch random patient"
          aria-label="Fetch random patient"
        >
          <span className="btn-icon" aria-hidden="true">↻</span>
          <span className="btn-text">{loading ? 'Fetching…' : 'New Patient'}</span>
        </button>
      </header>

      <div className="patient-details-grid">
        <DetailCard
          field="gender"
          label="Gender"
          rawValue={patient?.gender}
          values={patientDictionary.gender}
          onOpen={openValues}
        />
        <DetailCard
          field="race"
          label="Race"
          rawValue={patient?.race}
          values={patientDictionary.race}
          onOpen={openValues}
        />
        <DetailCard
          field="ethnicity"
          label="Ethnicity"
          rawValue={patient?.ethnicity}
          values={patientDictionary.ethnicity}
          onOpen={openValues}
        />
        <DetailCard
          field="birthplace"
          label="Birthplace"
          rawValue={patient?.birthplace}
          values={patientDictionary.birthplace}
          onOpen={openValues}
        />
        <DetailCard
          field="city"
          label="City"
          rawValue={patient?.city}
          values={patientDictionary.city}
          onOpen={openValues}
        />
        <DetailCard
          field="income"
          label="Income"
          rawValue={patient?.income}
          values={[]}
          editable
          onOpen={openValues}
        />
      </div>

      {openDetail && (
        <PatientValueMenu
          {...openDetail}
          onSelect={selectValue}
          onClose={() => setOpenDetail(null)}
        />
      )}
    </section>
  );
}