/**
 * API Service - Centralized HTTP client for DocBott backend.
 * All API calls go through this module for consistent error handling and auth.
 */

import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('docbott_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 responses globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('docbott_token');
      localStorage.removeItem('docbott_user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ──────────────────────────────────────────────
// Auth API
// ──────────────────────────────────────────────
export const authAPI = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => {
    const formData = new URLSearchParams();
    formData.append('username', data.username);
    formData.append('password', data.password);
    return api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
  },
  me: () => api.get('/auth/me'),
};

// ──────────────────────────────────────────────
// Documents API
// ──────────────────────────────────────────────
export const documentsAPI = {
  upload: (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress,
    });
  },
  list: () => api.get('/documents/'),
  get: (id) => api.get(`/documents/${id}`),
  delete: (id) => api.delete(`/documents/${id}`),
  search: (query) => api.get(`/documents/search/content?q=${encodeURIComponent(query)}`),
  analytics: () => api.get('/documents/analytics/overview'),
  relationships: () => api.get('/documents/relationships'),
  insights: (id) => api.get(`/documents/${id}/insights`),
  compare: (data) => api.post('/documents/compare', data),
  statsOverview: () => api.get('/documents/stats/overview'),
  ttsText: (id, page, maxChars) => {
    let url = `/documents/${id}/tts?max_chars=${maxChars || 5000}`;
    if (page != null) url += `&page=${page}`;
    return api.get(url);
  },
  annotations: (id) => api.get(`/documents/${id}/annotations`),
  addAnnotation: (id, data) => api.post(`/documents/${id}/annotations`, data),
  deleteAnnotation: (docId, annId) => api.delete(`/documents/${docId}/annotations/${annId}`),
  generateFlashcards: (data) => api.post('/documents/flashcards/generate', data),
};

// ──────────────────────────────────────────────
// Bookmarks API
// ──────────────────────────────────────────────
export const bookmarksAPI = {
  list: () => api.get('/bookmarks/'),
  create: (data) => api.post('/bookmarks/', data),
  delete: (id) => api.delete(`/bookmarks/${id}`),
};

// ──────────────────────────────────────────────
// Tags API
// ──────────────────────────────────────────────
export const tagsAPI = {
  list: (docId) => api.get(`/documents/${docId}/tags`),
  add: (docId, tag) => api.post(`/documents/${docId}/tags`, { tag }),
  remove: (docId, tagId) => api.delete(`/documents/${docId}/tags/${tagId}`),
};

// ──────────────────────────────────────────────
// Chat API
// ──────────────────────────────────────────────
export const chatAPI = {
  query: (data) => api.post('/chat/query', data),
  compare: (data) => api.post('/chat/compare', data),
  related: (data) => api.post('/chat/related', data),
  translate: (data) => api.post('/chat/translate', data),
  detectLanguage: (data) => api.post('/chat/detect-language', data),
  factCheck: (data) => api.post('/chat/fact-check', data),
  explainConfidence: (data) => api.post('/chat/explain-confidence', data),
  highlight: (data) => api.post('/chat/highlight', data),
  exportChat: (data) => api.post('/chat/export', data),
  sessions: () => api.get('/chat/sessions'),
  history: (sessionId) => api.get(`/chat/sessions/${sessionId}/history`),
  deleteSession: (sessionId) => api.delete(`/chat/sessions/${sessionId}`),
};

// ──────────────────────────────────────────────
// FAQ API
// ──────────────────────────────────────────────
export const faqAPI = {
  generate: (data) => api.post('/faqs/generate', data),
  list: (docId) => api.get(`/faqs/${docId ? `?document_id=${docId}` : ''}`),
  delete: (id) => api.delete(`/faqs/${id}`),
};

// ──────────────────────────────────────────────
// Reading Progress API
// ──────────────────────────────────────────────
export const progressAPI = {
  get: (docId) => api.get(`/progress/${docId}`),
  update: (docId, data) => api.put(`/progress/${docId}`, data),
  list: () => api.get('/progress/'),
};

// ──────────────────────────────────────────────
// User Preferences API
// ──────────────────────────────────────────────
export const preferencesAPI = {
  get: () => api.get('/preferences/'),
  update: (data) => api.put('/preferences/', data),
};

// ──────────────────────────────────────────────
// Feedback API
// ──────────────────────────────────────────────
export const feedbackAPI = {
  submit: (data) => api.post('/feedback/', data),
  stats: () => api.get('/feedback/stats'),
};

// ──────────────────────────────────────────────
// Admin API (requires admin role)
// ──────────────────────────────────────────────
export const adminAPI = {
  users: () => api.get('/admin/users'),
  updateRole: (userId, role) => api.put(`/admin/users/${userId}/role`, { role }),
  toggleActive: (userId) => api.put(`/admin/users/${userId}/toggle-active`),
  deleteUser: (userId) => api.delete(`/admin/users/${userId}`),
  stats: () => api.get('/admin/stats'),
};

export default api;
