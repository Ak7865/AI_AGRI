import axios from 'axios';

const API_URL = 'http://localhost:5000';

// Crop recommendation APIs
export const recommendCrops = (payload) =>
  axios.post(`${API_URL}/recommend_crops`, payload).then(res => res.data);

export const recommendFertilizer = (payload) =>
  axios.post(`${API_URL}/recommend_fertilizer`, payload).then(res => res.data);

// Pest detection APIs
export const detectPest = (file) => {
  const form = new FormData();
  form.append('image', file);
  return axios.post(`${API_URL}/detect_pest`, form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }).then(res => res.data);
};

export const detectPestDetailed = (file) => {
  const form = new FormData();
  form.append('image', file);
  return axios.post(`${API_URL}/detect_pest_detailed`, form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }).then(res => res.data);
};

export const getPestClasses = () =>
  axios.get(`${API_URL}/pest_classes`).then(res => res.data);

export const getModelInfo = () =>
  axios.get(`${API_URL}/model_info`).then(res => res.data);

export const getHealthStatus = () =>
  axios.get(`${API_URL}/health`).then(res => res.data);

// Utility function to get training visualizations
export const getTrainingVisualization = (filename) =>
  `${API_URL}/training_visualizations/${filename}`;