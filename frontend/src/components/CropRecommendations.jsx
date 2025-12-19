const CropRecommendations = ({ soilType, crops, onSelect, onBack }) => (
  <div className="panel">
    <div className="panel-heading">
      <div>
        <p className="eyebrow">Step 2</p>
        <h2>Select Preferred Crops</h2>
      </div>
      <p className="panel-subtitle">
        Based on the predicted soil profile <strong>{soilType || 'N/A'}</strong>, these crops are the best fit.
        Choose one to generate a fertilizer plan.
      </p>
    </div>

    <div className="card-grid">
      {crops.map((crop, index) => (
        <article key={crop} className="crop-card">
          <div className="crop-rank">#{index + 1}</div>
          <h3>{crop}</h3>
          <p>Highly compatible with your current soil and climate metrics.</p>
          <button className="btn ghost" onClick={() => onSelect(crop)}>
            Use {crop}
          </button>
        </article>
      ))}

      {crops.length === 0 && (
        <div className="empty-state">
          <p>No matching crops were found. Try adjusting the soil inputs.</p>
        </div>
      )}
    </div>

    <div className="panel-actions">
      <button className="btn secondary" onClick={onBack}>
        Adjust Soil Metrics
      </button>
    </div>
  </div>
);

export default CropRecommendations;


