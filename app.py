from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ============================================
# STATE NAME MAPPING
# ============================================
STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

def get_full_state_name(state_code):
    return STATE_NAMES.get(state_code, state_code)

def get_state_code(full_name):
    for code, name in STATE_NAMES.items():
        if name.lower() == full_name.lower():
            return code
    return full_name

# ============================================
# LOAD MODEL
# ============================================
model_path = r'C:\Users\Admin\Desktop\Projects\AIML\Prison Overcrowding Predictor\random_forest_model.pkl'
model = joblib.load(model_path)
print("✅ Model loaded!")

# ============================================
# LOAD FACILITY DATA
# ============================================
df = pd.read_csv(
    r"C:\Users\Admin\Desktop\Projects\AIML\Prison Overcrowding Predictor\Data\prisons.csv",
    encoding='latin1',
    on_bad_lines='skip',
    delimiter='\t'
)

# Filter and clean
display_df = df[df['STATUS'] == 'OPEN'].copy()
display_df = display_df[['FACILITYID', 'NAME', 'CITY', 'STATE', 'POPULATION', 'CAPACITY']]
display_df['CITY'] = display_df['CITY'].str.upper().str.strip()
display_df['STATE'] = display_df['STATE'].str.upper().str.strip()
display_df['NAME'] = display_df['NAME'].str.strip()
display_df['POPULATION'] = display_df['POPULATION'].replace(-999, np.nan)
display_df['CAPACITY'] = display_df['CAPACITY'].replace(-999, np.nan)
display_df = display_df.dropna()
display_df['POPULATION'] = display_df['POPULATION'].astype(int)
display_df['CAPACITY'] = display_df['CAPACITY'].astype(int)

states = sorted(display_df['STATE'].apply(get_full_state_name).unique())
print(f"✅ Loaded {len(display_df)} facilities")

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    return render_template('index.html', states=states)

@app.route('/get_cities', methods=['POST'])
def get_cities():
    data = request.get_json()
    state_full = data.get('state')
    if not state_full:
        return jsonify({'cities': []})
    state_code = get_state_code(state_full)
    cities = sorted(display_df[display_df['STATE'] == state_code]['CITY'].unique())
    return jsonify({'cities': cities})

@app.route('/get_facilities', methods=['POST'])
def get_facilities():
    data = request.get_json()
    state_full = data.get('state')
    city = data.get('city')
    if not state_full or not city:
        return jsonify({'facilities': []})
    state_code = get_state_code(state_full)
    filtered = display_df[(display_df['STATE'] == state_code) & (display_df['CITY'] == city)]
    
    facilities_list = []
    for _, row in filtered.iterrows():
        facilities_list.append({
            'id': int(row['FACILITYID']),
            'name': row['NAME'],
            'population': int(row['POPULATION']),
            'capacity': int(row['CAPACITY']),
            'available_slots': int(row['CAPACITY'] - row['POPULATION']),
            'is_overcrowded': int(row['POPULATION'] > row['CAPACITY'])
        })
    return jsonify({'facilities': facilities_list})

@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    try:
        data = request.get_json()
        facility_id = data.get('facility_id')
        population = float(data.get('population'))
        capacity = float(data.get('capacity'))
        
        facility = display_df[display_df['FACILITYID'] == int(facility_id)]
        facility_name = facility['NAME'].iloc[0] if not facility.empty else "Unknown"
        facility_city = facility['CITY'].iloc[0] if not facility.empty else "Unknown"
        facility_state = get_full_state_name(facility['STATE'].iloc[0]) if not facility.empty else "Unknown"
        
        # Make prediction
        features = np.array([[population, capacity]])
        prediction = model.predict(features)[0]
        
        available_slots = int(capacity - population)
        utilization = (population / capacity) * 100 if capacity > 0 else 0
        
        if prediction == 1:
            result = "OVERcrowded"
            result_class = "overcrowded"
        else:
            result = "NOT overcrowded"
            result_class = "safe"
        
        return jsonify({
            'success': True,
            'prediction': result,
            'result_class': result_class,
            'facility_name': facility_name,
            'facility_city': facility_city,
            'facility_state': facility_state,
            'facility_id': facility_id,
            'population': int(population),
            'capacity': int(capacity),
            'available_slots': available_slots,
            'utilization': f"{utilization:.1f}"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
    

if __name__ == '__main__':
    app.run(debug=True)
