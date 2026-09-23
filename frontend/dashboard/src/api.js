import axios from 'axios';
import { normalizeSchema } from './schema/schemaTypes';

const API_BASE_URL = 'http://localhost:8004'; // Replace with your actual backend URL


export const api = {
  // 1. Fetch DB schema (table names, column types, keys)
  /** @returns {Promise<import('./schema/schemaTypes').SchemaResponse>} */
  getSchema: async () => {
    const response = await axios.get(`${API_BASE_URL}/schema`);
    // Validate/normalize at the API boundary so every consumer receives the
    // same table/column model.
    return { tables: normalizeSchema(response.data) };
  },

  // 2. Get detailed patient record by ID
  getPatient: async (patientId) => {
    const response = await axios.get(`${API_BASE_URL}/get_patient`, {
      params: { id: patientId }
    });
    return response.data;
  },

  // 3. Load a random patient and their ordered event records
  getRandomPatient: async () => {
    const response = await axios.get(`${API_BASE_URL}/random_patient`);
    return response.data;
  },

  // 4. Run ML prediction given patient payload/features
  predict: async (patientFeatures) => {
    const response = await axios.post(`${API_BASE_URL}/predict`, patientFeatures);
    return response.data;
  }
};