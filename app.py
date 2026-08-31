import os
import glob
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from pybaseball import statcast_pitcher
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# 모델 저장 디렉터리 생성
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 8분할 코스 범주화 함수
def categorize_zone(row):
    px, pz = row['plate_x'], row['plate_z']
    sz_top, sz_bot = row['sz_top'], row['sz_bot']
    sz_mid = (sz_top + sz_bot) / 2

    if pd.isna(px) or pd.isna(pz) or pd.isna(sz_top) or pd.isna(sz_bot):
        return None

    # 스트라이크존 안쪽 (SZ)
    if -0.83 <= px <= 0.83 and sz_bot <= pz <= sz_top:
        if px <= 0:
            return 'SZ_UL' if pz >= sz_mid else 'SZ_LL'
        else:
            return 'SZ_UR' if pz >= sz_mid else 'SZ_LR'
    # 스트라이크존 바깥쪽 (CHASE)
    else:
        if px <= 0:
            return 'CHASE_UL' if pz >= sz_mid else 'CHASE_LL'
        else:
            return 'CHASE_UR' if pz >= sz_mid else 'CHASE_LR'


def train_and_save_models(pitcher_id):
    """
    메모리 사용량을 극소화한 Statcast 수집 및 경량 모델 학습 함수
    """
    # 1. 최근 1년치 데이터만 조회 (메모리 절약)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    try:
        df = statcast_pitcher(start_date, end_date, int(pitcher_id))
    except Exception as e:
        print(f"Statcast 수집 에러: {e}")
        return None, None

    if df is None or df.empty:
        return None, None

    # 2. 필수 컬럼만 선택하여 즉시 메모리 경량화
    req_cols = ['pitch_type', 'stand', 'balls', 'strikes', 'pitch_number', 'plate_x', 'plate_z', 'sz_top', 'sz_bot']
    df = df[req_cols].dropna().copy()

    # 시간순 정렬 (statcast는 최신순으로 넘어옴)
    df = df.iloc[::-1].reset_index(drop=True)

    # 8분할 존 피처 생성
    df['zone_target'] = df.apply(categorize_zone, axis=1)

    # 직전 투구 구종 (prev1_pitch) 파생 변수 생성
    df['prev1_pitch'] = df['pitch_type'].shift(1)
    df = df.dropna(subset=['prev1_pitch', 'zone_target']).copy()

    if len(df) < 50:  # 투구 데이터가 너무 적으면 학습 불가
        return None, None

    # 범주형 데이터 원-핫 인코딩
    df_encoded = pd.get_dummies(df, columns=['stand', 'prev1_pitch'])

    X = df_encoded.drop(columns=['pitch_type', 'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'zone_target'])
    y_pitch = df_encoded['pitch_type']
    y_zone = df_encoded['zone_target']

    # 3. 모델 경량화 (n_estimators=40, max_depth=8 로 제한하여 RAM 초과 방지)
    clf_pitch = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42, n_jobs=1)
    clf_pitch.fit(X, y_pitch)

    clf_zone = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42, n_jobs=1)
    clf_zone.fit(X, y_zone)

    # 학습에 사용된 컬럼 정보 포함하여 저장
    model_data_pitch = {'model': clf_pitch, 'columns': list(X.columns)}
    model_data_zone = {'model': clf_zone, 'columns': list(X.columns)}

    pitch_path = os.path.join(MODEL_DIR, f'pitch_{pitcher_id}.pkl')
    zone_path = os.path.join(MODEL_DIR, f'zone_{pitcher_id}.pkl')

    joblib.dump(model_data_pitch, pitch_path, compress=3)
    joblib.dump(model_data_zone, zone_path, compress=3)

    return model_data_pitch, model_data_zone


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        pitcher_id = str(data.get('pitcher_id')).strip()
        stand = data.get('stand')
        balls = int(data.get('balls'))
        strikes = int(data.get('strikes'))
        cum_pitch_count = int(data.get('cum_pitch_count'))
        prev1_pitch = data.get('prev1_pitch')

        pitch_path = os.path.join(MODEL_DIR, f'pitch_{pitcher_id}.pkl')
        zone_path = os.path.join(MODEL_DIR, f'zone_{pitcher_id}.pkl')

        # 기존 모델이 있으면 로드, 없으면 경량 학습 진행
        if os.path.exists(pitch_path) and os.path.exists(zone_path):
            model_data_pitch = joblib.load(pitch_path)
            model_data_zone = joblib.load(zone_path)
        else:
            # 오래된 다른 투수 모델 파일 삭제 (서버 용량 관리)
            old_files = glob.glob(os.path.join(MODEL_DIR, '*.pkl'))
            for f in old_files:
                try:
                    os.remove(f)
                except Exception:
                    pass

            model_data_pitch, model_data_zone = train_and_save_models(pitcher_id)

        if not model_data_pitch or not model_data_zone:
            return jsonify({'status': 'error', 'message': '선수 데이터를 불러올 수 없거나 투구 수가 적습니다.'}), 400

        # 예측용 입력 데이터 프레임 구축
        feature_cols = model_data_pitch['columns']
        input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

        # 수치형 값 채우기
        if 'balls' in input_df.columns: input_df.at[0, 'balls'] = balls
        if 'strikes' in input_df.columns: input_df.at[0, 'strikes'] = strikes
        if 'pitch_number' in input_df.columns: input_df.at[0, 'pitch_number'] = cum_pitch_count

        # 원-핫 인코딩 컬럼 매칭
        stand_col = f'stand_{stand}'
        prev_col = f'prev1_pitch_{prev1_pitch}'

        if stand_col in input_df.columns: input_df.at[0, stand_col] = 1
        if prev_col in input_df.columns: input_df.at[0, prev_col] = 1

        # 1. 구종 확률 예측
        clf_pitch = model_data_pitch['model']
        pitch_probs = clf_pitch.predict_proba(input_df)[0]
        pitch_classes = clf_pitch.classes_

        predictions = []
        for p_class, prob in zip(pitch_classes, pitch_probs):
            predictions.append({'pitch': p_class, 'prob': round(float(prob) * 100, 1)})
        predictions = sorted(predictions, key=lambda x: x['prob'], reverse=True)

        # 2. 8분할 코스 확률 예측
        clf_zone = model_data_zone['model']
        zone_probs = clf_zone.predict_proba(input_df)[0]
        zone_classes = clf_zone.classes_

        all_zones = ['SZ_UL', 'SZ_UR', 'SZ_LL', 'SZ_LR', 'CHASE_UL', 'CHASE_UR', 'CHASE_LL', 'CHASE_LR']
        zones_dict = {z: 0.0 for z in all_zones}

        for z_class, prob in zip(zone_classes, zone_probs):
            if z_class in zones_dict:
                zones_dict[z_class] = round(float(prob) * 100, 1)

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'zones': zones_dict
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
