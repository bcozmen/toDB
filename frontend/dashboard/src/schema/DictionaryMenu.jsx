import React, { useState, useMemo, useEffect, useRef } from 'react';

/**
 * Normalizes dictionary entries into an array of { code, label, category }.
 */
function normalizeEntries(tableName, dictionary, selectedKey) {
  if (!dictionary || typeof dictionary !== 'object') return [];

  const isPatient = tableName?.toLowerCase() === 'patients';

  if (isPatient) {
    if (selectedKey && dictionary[selectedKey]) {
      const values = Array.isArray(dictionary[selectedKey])
        ? dictionary[selectedKey]
        : [dictionary[selectedKey]];
      return values.map((val) => ({
        code: String(val),
        label: null,
        category: selectedKey,
      }));
    }

    // All patient categories combined if no specific key chosen
    return Object.entries(dictionary).flatMap(([cat, vals]) => {
      const items = Array.isArray(vals) ? vals : [vals];
      return items.map((val) => ({
        code: String(val),
        label: null,
        category: cat,
      }));
    });
  }

  // Standard code -> description dictionary
  return Object.entries(dictionary).map(([code, desc]) => ({
    code,
    label: typeof desc === 'string' ? desc : Array.isArray(desc) ? desc.join(', ') : String(desc ?? ''),
    category: null,
  }));
}

export function DictionaryMenu({
  tableName = '',
  dictionary = {},
  dictionaryKey = null,
  onClose,
}) {
  const [search, setSearch] = useState('');
  const [activeTab, setActiveTab] = useState(dictionaryKey);
  const [copiedCode, setCopiedCode] = useState(null);
  const searchInputRef = useRef(null);

  const isPatient = tableName?.toLowerCase() === 'patients';
  const patientCategories = useMemo(() => {
    if (!isPatient || !dictionary) return [];
    return Object.keys(dictionary);
  }, [isPatient, dictionary]);

  // If table is patients and no key was chosen initially, default tab to first category
  const currentCategory = isPatient ? (activeTab ?? patientCategories[0] ?? null) : null;

  const entries = useMemo(() => {
    return normalizeEntries(tableName, dictionary, currentCategory);
  }, [tableName, dictionary, currentCategory]);

  const filteredEntries = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter((e) =>
      e.code.toLowerCase().includes(q) || (e.label && e.label.toLowerCase().includes(q))
    );
  }, [entries, search]);

  // Handle Escape key to close
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose?.();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Auto focus search input on mount
  useEffect(() => {
    searchInputRef.current?.focus();
  }, []);

  const handleCopy = (code) => {
    navigator.clipboard?.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 1500);
  };

  const titleText = isPatient
    ? currentCategory
      ? `Patients • ${currentCategory.toUpperCase()} Values`
      : 'Patients • Value Dictionary'
    : `${tableName} • Code Dictionary`;

  return (
    <div className="schema-dict-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="schema-dict-card" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="schema-dict-header">
          <div className="schema-dict-header-left">
            <span className="schema-dict-kicker">DICTIONARY REFERENCE</span>
            <div className="schema-dict-title-row">
              <h3 className="schema-dict-title">{titleText}</h3>
              <span className="schema-dict-badge">
                {filteredEntries.length} {filteredEntries.length === 1 ? 'entry' : 'entries'}
              </span>
            </div>
          </div>
          <button
            type="button"
            className="schema-dict-close-btn"
            onClick={onClose}
            aria-label="Close dictionary"
          >
            ×
          </button>
        </div>

        {/* Patient Category Tabs (if patients table) */}
        {isPatient && patientCategories.length > 1 && (
          <div className="schema-dict-tabs">
            {patientCategories.map((cat) => (
              <button
                key={cat}
                type="button"
                className={`schema-dict-tab ${currentCategory === cat ? 'is-active' : ''}`}
                onClick={() => {
                  setActiveTab(cat);
                  setSearch('');
                }}
              >
                {cat}
                <span className="schema-dict-tab-count">
                  {Array.isArray(dictionary[cat]) ? dictionary[cat].length : 1}
                </span>
              </button>
            ))}
          </div>
        )}

        {/* Search Bar */}
        <div className="schema-dict-search-wrap">
          <svg className="schema-dict-search-icon" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
          </svg>
          <input
            ref={searchInputRef}
            type="text"
            className="schema-dict-search-input"
            placeholder={
              isPatient
                ? `Filter ${currentCategory || 'values'}…`
                : `Filter ${entries.length} codes or descriptions…`
            }
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button
              type="button"
              className="schema-dict-search-clear"
              onClick={() => setSearch('')}
              aria-label="Clear search"
            >
              ×
            </button>
          )}
        </div>

        {/* Entries List */}
        <div className="schema-dict-list">
          {filteredEntries.length === 0 ? (
            <div className="schema-dict-empty">
              {search ? (
                <>
                  No matches for <strong>"{search}"</strong>
                </>
              ) : (
                'No dictionary entries available.'
              )}
            </div>
          ) : (
            filteredEntries.map((entry, idx) => (
              <div key={`${entry.code}-${idx}`} className="schema-dict-item">
                <button
                  type="button"
                  className={`schema-dict-code-btn ${copiedCode === entry.code ? 'is-copied' : ''}`}
                  onClick={() => handleCopy(entry.code)}
                  title="Click to copy code"
                >
                  <span className="schema-dict-code">{entry.code}</span>
                  <span className="schema-dict-copy-hint">
                    {copiedCode === entry.code ? 'Copied!' : 'Copy'}
                  </span>
                </button>
                {entry.label ? (
                  <span className="schema-dict-desc">{entry.label}</span>
                ) : (
                  <span className="schema-dict-val-tag">Permitted Value</span>
                )}
              </div>
            ))
          )}
        </div>

        {/* Footer info */}
        <div className="schema-dict-footer">
          <span>Click any code to copy to clipboard</span>
          <span className="schema-dict-esc-hint">ESC to close</span>
        </div>
      </div>
    </div>
  );
}