import os, joblib, pandas as pd, numpy as np
from flask import Flask, render_template, request, jsonify
from pybaseball import statcast_pitcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

PITCH_TYPES = ['FF', 'SL', 'CH', 'CU', 'SI', 'ST', 'FS']
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
    # 포트폴리오 재현성을 위해 실험 데이터 기간을 고정 (정규시즌 기준)
    start_date = '2024-01-01'
    end_date = '2025-12-31'
    
    try:
        df = statcast_pitcher(start_date, end_date, int(pitcher_id))
    except:
        return None

    if df is None or df.empty: return None

    # 1. 정규시즌('R') 데이터만 필터링 및 시간순 정렬
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R'].copy()
        
    df = df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)

    # 2. 고유 경기(game_pk) 및 타석 기준 직전 구종 생성 (필터링 전 수행하여 시퀀스 보존)
    df['prev1_pitch'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1).fillna('FIRST')
    df['prev2_pitch'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(2).fillna('FIRST')

    # 3. 모델링 대상 구종 필터링 및 타자 맞춤형(sz_top/sz_bot) 동적 존 생성
    df = df[df['pitch_type'].isin(PITCH_TYPES)].copy()
    df['zone_target'] = df.apply(categorize_zone_dynamic, axis=1)
    
    # 4. 주자 상황(개별 ON/OFF) 처리 (중복 피처인 count_advantage 제거)
    for col in ['on_1b', 'on_2b', 'on_3b']:
        df[col] = df[col].notnull().astype(int) if col in df.columns else 0

    req_cols = [
        'pitch_type', 'zone_target', 'stand', 'balls', 'strikes', 
        'pitch_number', 'on_1b', 'on_2b', 'on_3b', 
        'prev1_pitch', 'prev2_pitch'
    ]
    df = df[req_cols].dropna().reset_index(drop=True)

    if len(df) < 50: return None

    # 원-핫 인코딩 (입력 데이터 포맷 고정)
    df_encoded = pd.get_dummies(df, columns=['stand', 'prev1_pitch', 'prev2_pitch'])
    
    X = df_encoded.drop(columns=['pitch_type', 'zone_target'])
    y_pitch = df_encoded['pitch_type']
    y_zone = df_encoded['zone_target']

    # 5. 시계열(과거->미래) 기준 80:20 분할 (Data Leakage 방지)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_pitch_train, y_pitch_test = y_pitch.iloc[:split_idx], y_pitch.iloc[split_idx:]
    y_zone_train, y_zone_test = y_zone.iloc[:split_idx], y_zone.iloc[split_idx:]

    # 모델 학습
    clf_pitch = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf_pitch.fit(X_train, y_pitch_train)
    
    clf_zone = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf_zone.fit(X_train, y_zone_train)

    # 6. 정밀 성능 평가 (Pitch Accuracy, Zone Accuracy 및 훈련 데이터 기반 엄밀한 베이스라인 산정)
    pitch_preds = clf_pitch.predict(X_test)
    zone_preds = clf_zone.predict(X_test)
    
    pitch_accuracy = round(float(accuracy_score(y_pitch_test, pitch_preds)) * 100, 1)
    zone_accuracy = round(float(accuracy_score(y_zone_test, zone_preds)) * 100, 1)
    
    # 훈련 데이터 최빈값 기준 엄격한 베이스라인 산정
    baseline_pitch = y_pitch_train.mode()[0]
    baseline_acc = round(float((y_pitch_test == baseline_pitch).mean()) * 100, 1)

    model_data = {
        'pitch_model': clf_pitch,
        'zone_model': clf_zone,
        'columns': list(X.columns),
        'accuracy': pitch_accuracy,
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

        # 캐싱된 모델 로드 또는 신규 학습
        if not os.path.exists(model_path):
            model_data = train_and_save_models(p_id)
        else:
            model_data = joblib.load(model_path)

        if not model_data:
            return jsonify({'status': 'error', 'message': '데이터가 부족하거나 수집 실패했습니다.'}), 400

        # 추론용 DataFrame 원패스 구축 (컬럼 정합성 유지)
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
            'zone_accuracy': model_data.get('zone_accuracy', 0.0),
            'baseline_acc': model_data.get('baseline_acc', 0.0)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
