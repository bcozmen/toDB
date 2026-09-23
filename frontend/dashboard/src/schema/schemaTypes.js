/**
 * @typedef {Object} ForeignKey
 * @property {string} table
 * @property {string} column
 */

/**
 * @typedef {Object} Column
 * @property {string} name
 * @property {string} type
 * @property {boolean} nullable
 * @property {boolean} pk
 * @property {ForeignKey|null} fk
 */

/**
 * @typedef {Object} Table
 * @property {string} name
 * @property {Column[]} columns
 * @property {Record<string, string|string[]>} dictionary
 */

/**
 * @typedef {Object} SchemaResponse
 * @property {Table[]} tables
 */

/**
 * Converts the API response into one stable frontend shape. This also accepts
 * the original raw schema.json shape, so the diagram does not depend on which
 * backend version is running.
 *
 * @param {SchemaResponse|Record<string, Record<string, Object>>} response
 * @returns {Table[]}
 */
export function normalizeSchema(response) {
  const tables = Array.isArray(response?.tables)
    ? response.tables
    : Object.entries(response ?? {}).map(([name, columns]) => ({
      name,
      dictionary: columns?.dictionary ?? {},
      columns,
    }));

  return tables.map((table) => ({
    name: table.name,
    dictionary: table.dictionary ?? {},
    columns: Array.isArray(table.columns)
      ? table.columns
        .filter((column) => column.name !== 'dictionary' && column.column_name !== 'dictionary')
        .map(normalizeColumn)
      : Object.entries(table.columns ?? {})
        .filter(([name]) => name !== 'dictionary')
        .map(([name, column]) =>
          normalizeColumn({ ...column, column_name: column.column_name ?? name })
        ),
  }));
}

/** @param {Object} column @returns {Column} */
function normalizeColumn(column) {
  // Accept both the backend shape (`foreign_key`) and an already-normalized
  // column (`fk`). This makes normalization idempotent.
  const foreignKey = column.foreign_key ?? column.fk;
  const fk = typeof foreignKey === 'string'
    ? parseForeignKey(foreignKey)
    : foreignKey && typeof foreignKey === 'object'
      ? { table: foreignKey.table, column: foreignKey.column }
      : null;

  return {
    name: column.name ?? column.column_name ?? '',
    type: column.type ?? column.column_type ?? 'UNKNOWN',
    nullable: column.nullable ?? column.null ?? true,
    pk: Boolean(column.pk ?? column.primary_key),
    fk,
  };
}

/** @param {string} value @returns {ForeignKey|null} */
function parseForeignKey(value) {
  const [table, column] = value.split('.', 2);
  return table && column ? { table, column } : null;
}
