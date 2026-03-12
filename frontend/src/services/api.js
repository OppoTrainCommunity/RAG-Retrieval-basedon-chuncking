import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
});

// Health & Stats
export const getHealth = () => api.get('/health');
export const getStats = () => api.get('/stats');

// CVs
export const uploadFiles = (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  return api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};
export const getCVs = () => api.get('/cvs');
export const getCV = (cvId) => api.get(`/cvs/${cvId}`);
export const deleteCV = (cvId) => api.delete(`/cvs/${cvId}`);

// Chat
export const sendChat = (query, model) => api.post('/chat', { query, model });
export const getModels = () => api.get('/models');

// Match
export const matchCandidates = (jobDescription, model, topK = 5) =>
  api.post('/match', { job_description: jobDescription, model, top_k: topK });

// Analytics
export const getSkillsAnalytics = () => api.get('/analytics/skills');
export const getExperienceAnalytics = () => api.get('/analytics/experience');

// Settings
export const saveApiKey = (apiKey) => api.post('/settings/api-key', { api_key: apiKey });
export const validateApiKey = (apiKey) => api.post('/settings/api-key/validate', { api_key: apiKey });

export default api;
