import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class SoilCropModel:
    def __init__(self, core_path):
        # Load dataset
        self.data = pd.read_csv(core_path)
        
        # Label Encode necessary categorical columns
        self.le_soil = LabelEncoder()
        self.le_crop = LabelEncoder()
        self.le_fertilizer = LabelEncoder()
        
        self.data['Soil Type Enc'] = self.le_soil.fit_transform(self.data['Soil Type'])
        self.data['Crop Type Enc'] = self.le_crop.fit_transform(self.data['Crop Type'])
        self.data['Fertilizer Name Enc'] = self.le_fertilizer.fit_transform(self.data['Fertilizer Name'])
        
        # Train soil type prediction model
        self.model_soil = RandomForestClassifier()
        X_soil = self.data[['Temparature', 'Humidity', 'Moisture']]
        y_soil = self.data['Soil Type Enc']
        self.model_soil.fit(X_soil, y_soil)

    def nutrient_level(self, val):
        # Simple thresholds: customize as needed
        if val < 10:
            return 'Low'
        elif val <= 30:
            return 'Medium'
        else:
            return 'High'

    def recommend_crops(self, Temparature, Humidity, Moisture, Nitrogen, Phosphorous, Potassium):
        # Predict soil type
        input_features = pd.DataFrame([{
            'Temparature': Temparature,
            'Humidity': Humidity,
            'Moisture': Moisture
        }])
        soil_enc = self.model_soil.predict(input_features)[0]
        soil_type = self.le_soil.inverse_transform([soil_enc])[0]

        # Classify nutrient levels
        nn = self.nutrient_level(Nitrogen)
        pp = self.nutrient_level(Phosphorous)
        kk = self.nutrient_level(Potassium)

        # Filter dataset by predicted soil type
        subset = self.data[self.data['Soil Type'] == soil_type]

        # Filter crops acceptable to nutrient levels
        def nutrient_match(row):
            r_n = self.nutrient_level(row['Nitrogen'])
            r_p = self.nutrient_level(row['Phosphorous'])
            r_k = self.nutrient_level(row['Potassium'])
            return r_n == nn and r_p == pp and r_k == kk

        matched = subset[subset.apply(nutrient_match, axis=1)]

        # Return unique crops or fallback
        recommended_crops = matched['Crop Type'].unique()
        if len(recommended_crops) == 0:
            recommended_crops = subset['Crop Type'].unique()

        return {
            'Soil Type': soil_type,
            'Nutrient Levels': {'Nitrogen': nn, 'Phosphorous': pp, 'Potassium': kk},
            'Recommended Crops': list(recommended_crops)
        }

    def describe_fertilizer_npk(self, fert_name):
        """
        Converts fertilizer names into human-readable description:
        - 3 part like '14-35-14': describe with highest nutrient
        - 2 part like '14-14': equal proportion description
        - others like 'Urea', 'DAP': return as is
        """
        if not fert_name or not isinstance(fert_name, str):
            return fert_name

        if '-' not in fert_name:
            return fert_name
        
        parts = fert_name.split('-')
        try:
            if len(parts) == 3:
                n, p, k = map(int, parts)
                max_val = max(n, p, k)
                if max_val == n:
                    high = 'nitrogen'
                elif max_val == p:
                    high = 'phosphorus'
                else:
                    high = 'potassium'
                return (f"NPK fertilizer high in {high}, with "
                        f"{n}% nitrogen, {p}% phosphorus, and {k}% potassium.")
            
            elif len(parts) == 2:
                n, p = map(int, parts)
                if n == p:
                    return (f"NPK fertilizer with equal proportions of nitrogen, phosphorus, "
                            f"and potassium, each {n}%.")
                else:
                    return (f"NPK fertilizer with {n}% nitrogen and {p}% phosphorus. "
                            "Potassium proportion not specified.")
            else:
                return fert_name
        except Exception:
            return fert_name
        
    def recommend_fertilizer(self, crop, n, p, k):
        # Validate crop label
        if crop not in self.le_crop.classes_:
            return {"error": "Crop not recognized"}
        crop_enc = self.le_crop.transform([crop])[0]
        fert_data = self.data[self.data['Crop Type Enc'] == crop_enc]
        if fert_data.empty:
            return {"error": "No fertilizer data for this crop"}
        
        # Find closest fertilizer match by Euclidean distance on NPK values
        fert_data = fert_data.copy()
        fert_data['distance'] = np.sqrt(
            (fert_data['Nitrogen'] - n) ** 2 +
            (fert_data['Phosphorous'] - p) ** 2 +
            (fert_data['Potassium'] - k) ** 2
        )
        best_row = fert_data.loc[fert_data['distance'].idxmin()]
        fert_name = self.le_fertilizer.inverse_transform([int(best_row['Fertilizer Name Enc'])])[0]
        description = self.describe_fertilizer_npk(fert_name)
        return {"Recommended Fertilizer": description}
