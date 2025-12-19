const FertilizerRecommendations = ({ recommendation, selectedCrop, onBack, onReset }) => {
  const message = recommendation?.['Recommended Fertilizer'] || recommendation?.error || 'No recommendation available';

  return (
    <div className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Step 3</p>
          <h2>Fertilizer Strategy</h2>
        </div>
        <p className="panel-subtitle">
          Tailored guidance for <strong>{selectedCrop}</strong>. Apply the recommended mix to balance soil nutrients.
        </p>
      </div>

      <div className="fertilizer-card">
        <h3>Recommendation</h3>
        <p>{message}</p>
      </div>

      <div className="panel-actions">
        <button className="btn secondary" onClick={onBack}>
          Back to Crop List
        </button>
        <button className="btn ghost" onClick={onReset}>
          Start Over
        </button>
      </div>
    </div>
  );
};

export default FertilizerRecommendations;


