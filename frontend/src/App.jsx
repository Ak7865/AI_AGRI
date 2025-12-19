import { useEffect, useState } from 'react';
import { Sprout, Bug } from 'lucide-react';
import SoilMetricsForm from './components/SoilMetricsForm';
import CropRecommendations from './components/CropRecommendations';
import FertilizerRecommendations from './components/FertilizerRecommendations';
import PestDetection from './components/PestDetection';
import { recommendCrops, recommendFertilizer, getPestClasses, getModelInfo } from './api';
import './App.css';

const INITIAL_METRICS = {
  temperature: '',
  humidity: '',
  moisture: '',
  nitrogen: '',
  phosphorous: '',
  potassium: '',
  ph: '',
  rainfall: '',
};

function App() {
  const [activeTab, setActiveTab] = useState('soil');
  const [currentStep, setCurrentStep] = useState('input');
  const [metrics, setMetrics] = useState(INITIAL_METRICS);
  const [soilType, setSoilType] = useState('');
  const [cropList, setCropList] = useState([]);
  const [selectedCrop, setSelectedCrop] = useState('');
  const [fertilizerRecommendation, setFertilizerRecommendation] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [pestClasses, setPestClasses] = useState([]);

  useEffect(() => {
    refreshPestMeta();
  }, []);

  const refreshPestMeta = async () => {
    try {
      const classesResponse = await getPestClasses();
      if (classesResponse.success) {
        setPestClasses(classesResponse.classes || []);
      }
    } catch (error) {
      console.error('Failed to load pest classes', error);
    }

    try {
      const info = await getModelInfo();
      setModelInfo(info);
    } catch (error) {
      console.error('Failed to load model info', error);
    }
  };

  const handleSoilSubmit = async (values) => {
    setMetrics(values);
    setStatus(null);
    setLoading(true);

    const payload = {
      Temparature: parseFloat(values.temperature),
      Humidity: parseFloat(values.humidity),
      Moisture: parseFloat(values.moisture),
      Nitrogen: parseFloat(values.nitrogen),
      Phosphorous: parseFloat(values.phosphorous),
      Potassium: parseFloat(values.potassium),
    };

    if (Object.values(payload).some((value) => Number.isNaN(value))) {
      setStatus({ type: 'error', message: 'Please provide all required soil metrics.' });
      setLoading(false);
      return;
    }

    try {
      const response = await recommendCrops(payload);
      setSoilType(response?.['Soil Type'] || 'Unknown');
      setCropList(response?.['Recommended Crops'] || []);
      setCurrentStep('crops');
      setStatus({ type: 'success', message: 'Crop recommendations ready.' });
    } catch (error) {
      console.error(error);
      setStatus({ type: 'error', message: 'Unable to fetch crop recommendations.' });
    } finally {
      setLoading(false);
    }
  };

  const handleCropSelection = async (crop) => {
    if (!crop) return;
    setStatus(null);
    setLoading(true);

    const nutrientPayload = {
      Nitrogen: parseFloat(metrics.nitrogen),
      Phosphorous: parseFloat(metrics.phosphorous),
      Potassium: parseFloat(metrics.potassium),
    };

    if (Object.values(nutrientPayload).some((value) => Number.isNaN(value))) {
      setStatus({ type: 'error', message: 'Please provide valid nutrient values.' });
      setLoading(false);
      return;
    }

    try {
      const response = await recommendFertilizer({ Crop: crop, ...nutrientPayload });
      setSelectedCrop(crop);
      setFertilizerRecommendation(response);
      setCurrentStep('fertilizer');
      setStatus({ type: 'success', message: 'Fertilizer guidance ready.' });
    } catch (error) {
      console.error(error);
      setStatus({ type: 'error', message: 'Unable to fetch fertilizer recommendation.' });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setMetrics(INITIAL_METRICS);
    setCropList([]);
    setSelectedCrop('');
    setFertilizerRecommendation(null);
    setCurrentStep('input');
    setStatus(null);
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">KRISHX</p>
          <h1>Intelligent Farming Recommendations</h1>
          <p>Combine soil analytics, fertilizer planning, and pest detection in one streamlined assistant.</p>
        </div>
      </header>

      <div className="tab-bar">
        <button
          className={activeTab === 'soil' ? 'active' : ''}
          onClick={() => setActiveTab('soil')}
        >
          <Sprout size={18} />
          Crop & Fertilizer
        </button>
        <button
          className={activeTab === 'pest' ? 'active' : ''}
          onClick={() => setActiveTab('pest')}
        >
          <Bug size={18} />
          Pest Detection
        </button>
      </div>

      {status && <div className={`status-banner ${status.type}`}>{status.message}</div>}
      {loading && <div className="status-banner loading">Processing request...</div>}

      {activeTab === 'soil' && (
        <>
          {currentStep === 'input' && (
            <SoilMetricsForm
              initialValues={metrics}
              onSubmit={handleSoilSubmit}
              loading={loading}
            />
          )}

          {currentStep === 'crops' && (
            <CropRecommendations
              soilType={soilType}
              crops={cropList}
              onSelect={handleCropSelection}
              onBack={() => setCurrentStep('input')}
            />
          )}

          {currentStep === 'fertilizer' && fertilizerRecommendation && (
            <FertilizerRecommendations
              recommendation={fertilizerRecommendation}
              selectedCrop={selectedCrop}
              onBack={() => setCurrentStep('crops')}
              onReset={handleReset}
            />
          )}
        </>
      )}

      {activeTab === 'pest' && (
        <PestDetection
          pestClasses={pestClasses}
          modelInfo={modelInfo}
          onRefreshMeta={refreshPestMeta}
        />
      )}
    </div>
  );
}

export default App;