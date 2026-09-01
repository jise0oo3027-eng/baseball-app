import os
import joblib
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify
from pybaseball import statcast_pitcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


app = Flask(__name__)

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 구종 설정
# ST(Sweeper)는 SL(Slider)로 통합
# FC(Cutter) 추가
# ============================================================
PITCH_TYPES = [
    'FF',   # Four-Seam Fastball
    'SL',   # Slider
    'CH',   # Changeup
    'CU',   # Curveball
    'KC',   # Knuckle Curve
    'SI',   # Sinker
    'FS',   # Splitter
    'FC'    # Cutter
]


# ============================================================
# 코스 8분할
# ============================================================
ALL_ZONES = [
    'SZ_UL', 'SZ_UR', 'SZ_LL', 'SZ_LR',
    'CHASE_UL', 'CHASE_UR', 'CHASE_LL', 'CHASE_LR'
]


# ============================================================
# 스트라이크존 8분할 함수
# ============================================================
def categorize_zone_dynamic(row):
    px = row['plate_x']
    pz = row['plate_z']
    top = row['sz_top']
    bot = row['sz_bot']

    if pd.isna(px) or pd.isna(pz) or pd.isna(top) or pd.isna(bot):
        return None

    # 스트라이크존 세로 중앙
    mid = (top + bot) / 2

    # 좌우 스트라이크존
    in_sz = (-0.83 <= px <= 0.83) and (bot <= pz <= top)

    # 스트라이크존 안/밖
    prefix = 'SZ_' if in_sz else 'CHASE_'

    # 좌우
    side = 'L' if px <= 0 else 'R'

    # 위/아래
    vert = 'U' if pz >= mid else 'L'

    return f"{prefix}{vert}{side}"


# ============================================================
# 모델 학습
# ============================================================
def train_and_save_models(pitcher_id):
    start_date = '2025-01-01'
    end_date = '2026-08-31'

    try:
        df = statcast_pitcher(
            start_date,
            end_date,
            int(pitcher_id)
        )
    except Exception as e:
        print(f"Statcast 수집 오류: {e}")
        return None

    if df is None or df.empty:
        return None

    # ========================================================
    # 1. 정규시즌만 사용
    # ========================================================
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R'].copy()

    if df.empty:
        return None

    # ========================================================
    # 2. 시간순 정렬
    # ========================================================
    df = df.sort_values([
        'game_date',
        'game_pk',
        'at_bat_number',
        'pitch_number'
    ]).reset_index(drop=True)

    # ========================================================
    # 3. ST → SL 통합
    # ========================================================
    if 'pitch_type' in df.columns:
        df['pitch_type'] = df['pitch_type'].replace(
            'ST',
            'SL'
        )

    # ========================================================
    # 4. 이전 구종 생성
    # 필터링 전에 생성해야 실제 투구 흐름을 유지할 수 있음
    # ========================================================
    df['prev1_pitch'] = (
        df.groupby(['game_pk', 'at_bat_number'])['pitch_type']
        .shift(1)
        .fillna('FIRST')
    )

    df['prev2_pitch'] = (
        df.groupby(['game_pk', 'at_bat_number'])['pitch_type']
        .shift(2)
        .fillna('FIRST')
    )

    # ========================================================
    # 5. 모델링 대상 구종만 사용
    # ========================================================
    df = df[
        df['pitch_type'].isin(PITCH_TYPES)
    ].copy()

    if df.empty:
        return None

    # ========================================================
    # 6. 동적 스트라이크존 생성
    # ========================================================
    df['zone_target'] = df.apply(
        categorize_zone_dynamic,
        axis=1
    )

    # ========================================================
    # 7. 주자 상황
    # ========================================================
    for col in ['on_1b', 'on_2b', 'on_3b']:
        if col in df.columns:
            df[col] = (
                df[col]
                .notnull()
                .astype(int)
            )
        else:
            df[col] = 0

    # ========================================================
    # 8. 아웃 카운트 (outs_when_up)
    # ========================================================
    if 'outs_when_up' in df.columns:
        df['outs'] = df['outs_when_up']
    else:
        df['outs'] = 0

    # ========================================================
    # 9. 이닝
    # ========================================================
    if 'inning' in df.columns:
        df['inning'] = df['inning'].fillna(0)
    else:
        df['inning'] = 0

    # ========================================================
    # 10. 경기 투구 수
    # ========================================================
    df['game_pitch_count'] = (
        df.groupby('game_pk')
        .cumcount()
    )

    # ========================================================
    # 11. 점수 차 (투수팀 기준)
    # ========================================================
    if (
        'home_score' in df.columns and
        'away_score' in df.columns and
        'inning_topbot' in df.columns
    ):
        df['home_score'] = df['home_score'].fillna(0)
        df['away_score'] = df['away_score'].fillna(0)

        df['score_diff'] = np.where(
            df['inning_topbot'].str.lower().eq('top'),
            df['away_score'] - df['home_score'],
            df['home_score'] - df['away_score']
        )
    else:
        df['score_diff'] = 0

    # ========================================================
    # 12. 필요한 컬럼 (pitch_number 제거 완료)
    # ========================================================
    req_cols = [
        'pitch_type',
        'zone_target',
        'stand',
        'balls',
        'strikes',
        'inning',
        'score_diff',
        'game_pitch_count',
        'outs',
        'on_1b',
        'on_2b',
        'on_3b',
        'prev1_pitch',
        'prev2_pitch'
    ]

    for col in req_cols:
        if col not in df.columns:
            print(f"필요 컬럼 없음: {col}")
            return None

    df = (
        df[req_cols]
        .dropna()
        .reset_index(drop=True)
    )

    if len(df) < 50:
        print(f"데이터 부족: {len(df)} pitches")
        return None

    # ========================================================
    # 13. One-Hot Encoding
    # ========================================================
    df_encoded = pd.get_dummies(
        df,
        columns=[
            'stand',
            'prev1_pitch',
            'prev2_pitch'
        ]
    )

    # ========================================================
    # 14. 입력 / 출력 데이터 분리
    # ========================================================
    X = df_encoded.drop(
        columns=[
            'pitch_type',
            'zone_target'
        ]
    )

    y_pitch = df_encoded['pitch_type']
    y_zone = df_encoded['zone_target']

    # ========================================================
    # 15. 시간순 80 : 20 분할
    # ========================================================
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_pitch_train = y_pitch.iloc[:split_idx]
    y_pitch_test = y_pitch.iloc[split_idx:]

    y_zone_train = y_zone.iloc[:split_idx]
    y_zone_test = y_zone.iloc[split_idx:]

    # ========================================================
    # 16. 구종 예측 Random Forest
    # ========================================================
    clf_pitch = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf_pitch.fit(X_train, y_pitch_train)

    # ========================================================
    # 17. 코스 예측 Random Forest
    # ========================================================
    clf_zone = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf_zone.fit(X_train, y_zone_train)

    # ========================================================
    # 18. Top-1 Accuracy
    # ========================================================
    pitch_preds = clf_pitch.predict(X_test)
    pitch_accuracy = round(
        float(accuracy_score(y_pitch_test, pitch_preds)) * 100,
        1
    )

    # ========================================================
    # 19. Top-2 Accuracy
    # ========================================================
    pitch_probs = clf_pitch.predict_proba(X_test)
    classes = clf_pitch.classes_

    top2_indices = np.argsort(pitch_probs, axis=1)[:, -2:]
    top2_correct = sum(
        1 for i, true_label in enumerate(y_pitch_test)
        if true_label in classes[top2_indices[i]]
    )
    top2_accuracy = round(
        float(top2_correct / len(y_pitch_test)) * 100,
        1
    )

    # ========================================================
    # 20. 코스 Accuracy
    # ========================================================
    zone_preds = clf_zone.predict(X_test)
    zone_accuracy = round(
        float(accuracy_score(y_zone_test, zone_preds)) * 100,
        1
    )

    # ========================================================
    # 21. 최빈구종 Baseline
    # ========================================================
    baseline_pitch = y_pitch_train.mode()[0]
    baseline_acc = round(
        float((y_pitch_test == baseline_pitch).mean()) * 100,
        1
    )

    # ========================================================
    # 22. 모델 정보 저장
    # ========================================================
    model_data = {
        'pitch_model': clf_pitch,
        'zone_model': clf_zone,
        'columns': list(X.columns),
        'accuracy': pitch_accuracy,
        'top2_accuracy': top2_accuracy,
        'zone_accuracy': zone_accuracy,
        'baseline_acc': baseline_acc,
        'baseline_pitch': baseline_pitch,
        'pitch_types': PITCH_TYPES
    }

    # ========================================================
    # 23. 모델 파일 저장
    # ========================================================
    model_path = os.path.join(MODEL_DIR, f'{pitcher_id}.pkl')
    joblib.dump(model_data, model_path, compress=3)

    print("\n===== 모델 학습 완료 =====")
    print(f"투수 ID: {pitcher_id} | 데이터 수: {len(df)}")
    print(
        f"Baseline: {baseline_acc}% | Top-1: {pitch_accuracy}% | "
        f"Top-2: {top2_accuracy}% | Zone: {zone_accuracy}%"
    )

    return model_data


# ============================================================
# 메인 페이지
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


# ============================================================
# 예측 API
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 기본 입력 (pitch_number 제거 완료)
        p_id = str(data.get('pitcher_id')).strip()
        balls = int(data.get('balls', 0))
        strikes = int(data.get('strikes', 0))

        # 경기 상황
        inning = int(data.get('inning', 1))
        score_diff = int(data.get('score_diff', 0))
        game_pitch_count = int(data.get('game_pitch_count', 0))
        outs = int(data.get('outs', 0))

        # 주자
        on_1b = int(data.get('on_1b', 0))
        on_2b = int(data.get('on_2b', 0))
        on_3b = int(data.get('on_3b', 0))

        # 타자 및 이전 구종
        stand = data.get('stand')
        prev1_pitch = data.get('prev1_pitch')
        prev2_pitch = data.get('prev2_pitch')

        # ST → SL 변환
        if prev1_pitch == 'ST':
            prev1_pitch = 'SL'
        if prev2_pitch == 'ST':
            prev2_pitch = 'SL'

        # 모델 불러오기 또는 학습
        model_path = os.path.join(MODEL_DIR, f'{p_id}.pkl')
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
        else:
            model_data = train_and_save_models(p_id)

        if not model_data:
            return jsonify({
                'status': 'error',
                'message': '데이터가 부족하거나 수집에 실패했습니다.'
            }), 400

        # 입력 DataFrame 생성
        cols = model_data['columns']
        input_df = pd.DataFrame(0, index=[0], columns=cols)

        # 수치형 변수 매핑 (pitch_number 제거 완료)
        numeric_features = [
            ('balls', balls),
            ('strikes', strikes),
            ('inning', inning),
            ('score_diff', score_diff),
            ('game_pitch_count', game_pitch_count),
            ('outs', outs),
            ('on_1b', on_1b),
            ('on_2b', on_2b),
            ('on_3b', on_3b)
        ]

        for col, val in numeric_features:
            if col in input_df.columns:
                input_df.at[0, col] = val

        # One-Hot Encoding 컬럼 매칭
        one_hot_features = [
            f'stand_{stand}',
            f'prev1_pitch_{prev1_pitch}',
            f'prev2_pitch_{prev2_pitch}'
        ]

        for col_name in one_hot_features:
            if col_name in input_df.columns:
                input_df.at[0, col_name] = 1

        clf_pitch = model_data['pitch_model']
        clf_zone = model_data['zone_model']

        # 구종 예측
        pitch_probs = clf_pitch.predict_proba(input_df)[0]
        pitch_classes = clf_pitch.classes_

        predictions = [
            {
                'pitch': p_class,
                'prob': round(float(prob) * 100, 1)
            }
            for p_class, prob in zip(pitch_classes, pitch_probs)
        ]
        predictions = sorted(predictions, key=lambda x: x['prob'], reverse=True)

        # 코스 예측
        zone_probs = clf_zone.predict_proba(input_df)[0]
        zone_classes = clf_zone.classes_

        zone_map = {z: 0.0 for z in ALL_ZONES}
        for zone_class, prob in zip(zone_classes, zone_probs):
            if zone_class in zone_map:
                zone_map[zone_class] = round(float(prob) * 100, 1)

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'zones': zone_map,
            'accuracy': model_data.get('accuracy', 0.0),
            'top2_accuracy': model_data.get('top2_accuracy', 0.0),
            'zone_accuracy': model_data.get('zone_accuracy', 0.0),
            'baseline_acc': model_data.get('baseline_acc', 0.0)
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================
# 실행
# ============================================================
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
