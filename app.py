import os, joblib, pandas as pd, numpy as np
from flask import Flask, render_template, request, jsonify
from pybaseball import statcast_pitcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

PITCH_TYPES = ['FF', 'SL', 'CH', 'CU', 'SI', 'FS', 'FC']
ALL_ZONES = ['SZ_UL', 'SZ_UR', 'SZ_LL', 'SZ_LR', 'CHASE_UL', 'CHASE_UR', 'CHASE_LL', 'CHASE_LR']

def categorize_zone_dynamic(row):
    px, pz = row['plate_x'], row['plate_z']
    top, bot = row['sz_top'], row['sz_bot']
    
    if pd.isna(px) or pd.isna(pz) or pd.isna(top) or pd.isna(bot): 
        return None
        
    mid = (top + bot) / 2
    in_sz = -0.83 <= px <= 0.83 and bot <= pz <= top
    
    prefix = 'SZ_' if in_sz else 'CHASE_'
    side = 'L' if px <= 0 else 'R'
    vert = 'U' if pz >= mid else 'L'
    return f"{prefix}{vert}{side}"

def train_and_save_models(pitcher_id):
    start_date = '2024-01-01'
    end_date = '2025-12-31'
    
    try:
        df = statcast_pitcher(start_date, end_date, int(pitcher_id))
    except:
        return None

    if df is None or df.empty: return None

    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R'].copy()
        
    df = df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)

    if 'pitch_type' in df.columns:
        df['pitch_type'] = df['pitch_type'].replace('ST', 'SL')

    df['prev1_pitch'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1).fillna('FIRST')
    df['prev2_pitch'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(2).fillna('FIRST')

    df = df[df['pitch_type'].isin(PITCH_TYPES)].copy()
    df['zone_target'] = df.apply(categorize_zone_dynamic, axis=1)
    
    for col in ['on_1b', 'on_2b', 'on_3b']:
        df[col] = df[col].notnull().astype(int) if col in df.columns else 0

    req_cols = [
        'pitch_type', 'zone_target', 'stand', 'balls', 'strikes', 
        'pitch_number', 'on_1b', 'on_2b', 'on_3b', 
        'prev1_pitch', 'prev2_pitch'
    ]
    df = df[req_cols].dropna().reset_index(drop=True)

    if len(df) < 50: return None

    df_encoded = pd.get_dummies(df, columns=['stand', 'prev1_pitch', 'prev2_pitch'])
    
    X = df_encoded.drop(columns=['pitch_type', 'zone_target'])
    y_pitch = df_encoded['pitch_type']
    y_zone = df_encoded['zone_target']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_pitch_train, y_pitch_test = y_pitch.iloc[:split_idx], y_pitch.iloc[split_idx:]
    y_zone_train, y_zone_test = y_zone.iloc[:split_idx], y_zone.iloc[split_idx:]

    clf_pitch = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_pitch.fit(X_train, y_pitch_train)
    
    clf_zone = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_zone.fit(X_train, y_zone_train)

    # 기본 성능 평가
    pitch_preds = clf_pitch.predict(X_test)
    zone_preds = clf_zone.predict(X_test)
    
    pitch_accuracy = round(float(accuracy_score(y_pitch_test, pitch_preds)) * 100, 1)
    zone_accuracy = round(float(accuracy_score(y_zone_test, zone_preds)) * 100, 1)
    
    # 🎯 Top-2 Accuracy 계산 로직 추가
    pitch_probs = clf_pitch.predict_proba(X_test)
    classes = clf_pitch.classes_
    top2_correct = 0
    
    for i, true_label in enumerate(y_pitch_test):
        # 확률이 높은 순으로 상위 2개 클래스 인덱스 추출
        top2_indices = pitch_probs[i].argsort()[-2:][::-1]
        top2_classes = [classes[idx] for idx in top2_indices]
        if true_label in top2_classes:
            top2_correct += 1
            
    top2_accuracy = round(float(top2_correct / len(y_pitch_test)) * 100, 1)

    baseline_pitch = y_pitch_train.mode()[0]
    baseline_acc = round(float((y_pitch_test == baseline_pitch).mean()) * 100, 1)

    model_data = {
        'pitch_model': clf_pitch,
        'zone_model': clf_zone,
        'columns': list(X.columns),
        'accuracy': pitch_accuracy,
        'top2_accuracy': top2_accuracy, # 추가된 Top-2 지표 저장
        'zone_accuracy': zone_accuracy,
        'baseline_acc': baseline_acc
    }
    
    joblib.dump(model_data, os.path.join(MODEL_DIR, f'{pitcher_id}.pkl'), compress=3)
    return model_data


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        p_id = str(data.get('pitcher_id')).strip()
        model_path = os.path.join(MODEL_DIR, f'{p_id}.pkl')

        if not os.path.exists(model_path):
            model_data = train_and_save_models(p_id)
        else:
            model_data = joblib.load(model_path)

        if not model_data:
            return jsonify({'status': 'error', 'message': '데이터가 부족하거나 수집 실패했습니다.'}), 400

        cols = model_data['columns']
        input_df = pd.DataFrame(0, index=[0], columns=cols)
        
        balls = int(data.get('balls', 0))
        strikes = int(data.get('strikes', 0))
        
        vals = {
            'balls': balls,
            'strikes': strikes,
            'pitch_number': int(data.get('at_bat_pitch_number', 1)),
            'on_1b': int(data.get('on_1b', 0)),
            'on_2b': int(data.get('on_2b', 0)),
            'on_3b': int(data.get('on_3b', 0)),
            f"stand_{data.get('stand')}": 1,
            f"prev1_pitch_{data.get('prev1_pitch')}": 1,
            f"prev2_pitch_{data.get('prev2_pitch')}": 1
        }

        for k, v in vals.items():
            if k in input_df.columns:
                input_df.at[0, k] = v

        clf_pitch, clf_zone = model_data['pitch_model'], model_data['zone_model']
        
        preds = sorted([
            {'pitch': c, 'prob': round(float(p) * 100, 1)} 
            for c, p in zip(clf_pitch.classes_, clf_pitch.predict_proba(input_df)[0])
        ], key=lambda x: x['prob'], reverse=True)

        zone_map = {z: 0.0 for z in ALL_ZONES}
        zone_map.update({c: round(float(p) * 100, 1) for c, p in zip(clf_zone.classes_, clf_zone.predict_proba(input_df)[0]) if c in zone_map})

        return jsonify({
            'status': 'success',
            'predictions': preds,
            'zones': zone_map,
            'accuracy': model_data.get('accuracy', 0.0),
            'top2_accuracy': model_data.get('top2_accuracy', 0.0), # 프론트엔드로 전달
            'zone_accuracy': model_data.get('zone_accuracy', 0.0),
            'baseline_acc': model_data.get('baseline_acc', 0.0)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
