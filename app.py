import os
import joblib
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify
from pybaseball import statcast_pitcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


app = Flask(__name__)

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 구종 설정
# ST(Sweeper)는 SL(Slider)로 통합
# ============================================================
PITCH_TYPES = [
    'FF',
    'SL',
    'CH',
    'CU',
    'KC',
    'SI',
    'FS',
    'FC'
]


# ============================================================
# MLB Statcast 공식 Zone
# 1~9  : Strike Zone 3x3
# 11~14: Chase Zone 4분할
# ============================================================
ALL_ZONES = list(range(1, 10)) + [11, 12, 13, 14]


# ============================================================
# Zone 표시용 이름
# ============================================================
ZONE_NAMES = {
    1: 'SZ_1',
    2: 'SZ_2',
    3: 'SZ_3',
    4: 'SZ_4',
    5: 'SZ_5',
    6: 'SZ_6',
    7: 'SZ_7',
    8: 'SZ_8',
    9: 'SZ_9',
    11: 'CHASE_11',
    12: 'CHASE_12',
    13: 'CHASE_13',
    14: 'CHASE_14'
}


# ============================================================
# Feature Importance
# ============================================================
def extract_top_features(model, columns, top_n=10):
    importances = model.feature_importances_
    ranked = sorted(zip(columns, importances), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        {
            'feature': name,
            'importance': round(float(score), 4)
        }
        for name, score in ranked
    ]


# ============================================================
# 모델 학습
# ============================================================
def train_and_save_models(pitcher_id):
    start_date = '2025-01-01'
    end_date = '2026-08-31'

    try:
        df = statcast_pitcher(start_date, end_date, int(pitcher_id))
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
    # 3. ST → SL
    # ========================================================
    df['pitch_type'] = df['pitch_type'].replace('ST', 'SL')

    # ========================================================
    # 4. 이전 구종
    # 필터링 전에 생성
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
    # 5. 모델링 대상 구종
    # ========================================================
    df = df[df['pitch_type'].isin(PITCH_TYPES)].copy()

    if df.empty:
        return None

    # ========================================================
    # 6. 공식 Statcast Zone 사용
    # 1~9 + 11~14
    # ========================================================
    df['zone_target'] = pd.to_numeric(df['zone'], errors='coerce')

    df = df[df['zone_target'].isin(ALL_ZONES)].copy()

    if df.empty:
        return None

    # ========================================================
    # 7. 주자 상황
    # ========================================================
    for col in ['on_1b', 'on_2b', 'on_3b']:
        if col in df.columns:
            df[col] = df[col].notna().astype(int)
        else:
            df[col] = 0

    # ========================================================
    # 8. 아웃 카운트
    # ========================================================
    if 'outs_when_up' in df.columns:
        df['outs'] = df['outs_when_up'].fillna(0)
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
    df['game_pitch_count'] = df.groupby('game_pk').cumcount()

    # ========================================================
    # 11. 점수 차
    # 투수팀 기준
    # ========================================================
    if all(col in df.columns for col in ['home_score', 'away_score', 'inning_topbot']):
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
    # 12. 필요한 컬럼
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

    # ========================================================
    # 13. 핵심 결측치만 제거
    # pitch_type / zone_target / stand
    # ========================================================
    df = df.dropna(
        subset=['pitch_type', 'zone_target', 'stand']
    ).copy()

    # ========================================================
    # 나머지 결측치는 수치형 기준으로 보정
    # ========================================================
    numeric_cols = [
        'balls',
        'strikes',
        'inning',
        'score_diff',
        'game_pitch_count',
        'outs',
        'on_1b',
        'on_2b',
        'on_3b'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.reset_index(drop=True)

    if len(df) < 50:
        print(f"데이터 부족: {len(df)} pitches")
        return None

    # ========================================================
    # 14. One-Hot Encoding
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
    # 15. 입력 / 출력
    # ========================================================
    X = df_encoded.drop(
        columns=['pitch_type', 'zone_target']
    )

    y_pitch = df_encoded['pitch_type']
    y_zone = df_encoded['zone_target'].astype(int)

    # ========================================================
    # 16. 경기 단위 시간순 80:20 분할
    # ========================================================
    games = df['game_pk'].drop_duplicates().tolist()

    split_game_idx = int(len(games) * 0.8)

    train_games = set(games[:split_game_idx])
    test_games = set(games[split_game_idx:])

    train_mask = df['game_pk'].isin(train_games)
    test_mask = df['game_pk'].isin(test_games)

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_pitch_train = y_pitch.loc[train_mask]
    y_pitch_test = y_pitch.loc[test_mask]

    y_zone_train = y_zone.loc[train_mask]
    y_zone_test = y_zone.loc[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        print("Train/Test 데이터가 부족합니다.")
        return None

    # ========================================================
    # 17. 구종 예측 모델
    # ========================================================
    clf_pitch = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    clf_pitch.fit(X_train, y_pitch_train)

    # ========================================================
    # 18. 구종별 코스 모델
    #
    # 각각 P(zone | pitch, 상황)을 학습
    # ========================================================
    zone_models = {}

    for pitch in PITCH_TYPES:
        pitch_mask = train_mask & (df['pitch_type'] == pitch)

        X_pitch = X.loc[pitch_mask]
        y_zone_pitch = y_zone.loc[pitch_mask]

        if len(X_pitch) < 20:
            print(f"{pitch}: 코스 학습 데이터 부족 ({len(X_pitch)})")
            continue

        if y_zone_pitch.nunique() < 2:
            print(f"{pitch}: 코스 종류가 1개뿐이라 모델 생성 생략")
            continue

        zone_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        zone_model.fit(X_pitch, y_zone_pitch)
        zone_models[pitch] = zone_model

    # ========================================================
    # 19. 구종 Top-1
    # ========================================================
    pitch_preds = clf_pitch.predict(X_test)

    pitch_accuracy = round(
        accuracy_score(y_pitch_test, pitch_preds) * 100,
        1
    )

    # ========================================================
    # 20. 구종 Top-2
    # ========================================================
    pitch_probs_test = clf_pitch.predict_proba(X_test)
    pitch_classes = clf_pitch.classes_

    top2_indices = np.argsort(
        pitch_probs_test,
        axis=1
    )[:, -2:]

    top2_correct = sum(
        1
        for i, true_label in enumerate(y_pitch_test)
        if true_label in pitch_classes[top2_indices[i]]
    )

    top2_accuracy = round(
        top2_correct / len(y_pitch_test) * 100,
        1
    )

    # ========================================================
    # 21. 구종 Macro F1
    # ========================================================
    pitch_macro_f1 = round(
        f1_score(
            y_pitch_test,
            pitch_preds,
            average='macro',
            zero_division=0
        ) * 100,
        1
    )

    # ========================================================
    # 22. 코스 예측
    #
    # P(zone) = Σ P(pitch) × P(zone | pitch)
    # ========================================================
    combined_zone_probs = np.zeros(
        (len(X_test), len(ALL_ZONES)),
        dtype=float
    )

    zone_index = {
        zone: idx
        for idx, zone in enumerate(ALL_ZONES)
    }

    for pitch_idx, pitch in enumerate(pitch_classes):
        if pitch not in zone_models:
            continue

        pitch_probability = pitch_probs_test[:, pitch_idx]

        zone_model = zone_models[pitch]
        zone_probs = zone_model.predict_proba(X_test)

        for class_idx, zone_class in enumerate(zone_model.classes_):
            if zone_class in zone_index:
                combined_zone_probs[
                    :,
                    zone_index[zone_class]
                ] += (
                    pitch_probability *
                    zone_probs[:, class_idx]
                )

    # ========================================================
    # 23. 코스 Top-1
    # ========================================================
    zone_pred_indices = np.argmax(
        combined_zone_probs,
        axis=1
    )

    zone_preds = np.array([
        ALL_ZONES[idx]
        for idx in zone_pred_indices
    ])

    zone_accuracy = round(
        accuracy_score(y_zone_test, zone_preds) * 100,
        1
    )

    # ========================================================
    # 24. 코스 Macro F1
    # ========================================================
    zone_macro_f1 = round(
        f1_score(
            y_zone_test,
            zone_preds,
            labels=ALL_ZONES,
            average='macro',
            zero_division=0
        ) * 100,
        1
    )

    # ========================================================
    # 25. 최빈구종 Baseline
    # ========================================================
    baseline_pitch = y_pitch_train.mode()[0]

    baseline_acc = round(
        (y_pitch_test == baseline_pitch).mean() * 100,
        1
    )

    # ========================================================
    # 26. Feature Importance
    # ========================================================
    pitch_feature_importance = extract_top_features(
        clf_pitch,
        X.columns,
        top_n=10
    )

    # ========================================================
    # 27. 모델 저장
    # ========================================================
    model_data = {
        'pitch_model': clf_pitch,
        'zone_models': zone_models,
        'columns': list(X.columns),
        'accuracy': pitch_accuracy,
        'top2_accuracy': top2_accuracy,
        'zone_accuracy': zone_accuracy,
        'pitch_macro_f1': pitch_macro_f1,
        'zone_macro_f1': zone_macro_f1,
        'baseline_acc': baseline_acc,
        'baseline_pitch': baseline_pitch,
        'pitch_types': PITCH_TYPES,
        'all_zones': ALL_ZONES,
        'zone_names': ZONE_NAMES
    }

    model_path = os.path.join(
        MODEL_DIR,
        f'{pitcher_id}.pkl'
    )

    joblib.dump(
        model_data,
        model_path,
        compress=3
    )

    # ========================================================
    # 28. 결과 출력
    # ========================================================
    print("\n===== 모델 학습 완료 =====")
    print(
        f"투수 ID: {pitcher_id} | "
        f"데이터 수: {len(df)} | "
        f"경기 수: {len(games)}"
    )

    print(
        f"Baseline: {baseline_acc}% | "
        f"Pitch Top-1: {pitch_accuracy}% | "
        f"Pitch Top-2: {top2_accuracy}% | "
        f"Zone: {zone_accuracy}%"
    )

    print(
        f"Pitch Macro F1: {pitch_macro_f1} | "
        f"Zone Macro F1: {zone_macro_f1}"
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

        p_id = str(data.get('pitcher_id')).strip()

        balls = int(data.get('balls', 0))
        strikes = int(data.get('strikes', 0))

        inning = int(data.get('inning', 1))
        score_diff = int(data.get('score_diff', 0))
        game_pitch_count = int(data.get('game_pitch_count', 0))
        outs = int(data.get('outs', 0))

        on_1b = int(data.get('on_1b', 0))
        on_2b = int(data.get('on_2b', 0))
        on_3b = int(data.get('on_3b', 0))

        stand = data.get('stand', 'R')
        prev1_pitch = data.get('prev1_pitch', 'FIRST')
        prev2_pitch = data.get('prev2_pitch', 'FIRST')

        # ST → SL
        if prev1_pitch == 'ST':
            prev1_pitch = 'SL'

        if prev2_pitch == 'ST':
            prev2_pitch = 'SL'

        # ====================================================
        # 모델 불러오기 또는 학습
        # ====================================================
        model_path = os.path.join(
            MODEL_DIR,
            f'{p_id}.pkl'
        )

        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
        else:
            model_data = train_and_save_models(p_id)

        if not model_data:
            return jsonify({
                'status': 'error',
                'message': '데이터가 부족하거나 수집에 실패했습니다.'
            }), 400

        # ====================================================
        # 입력 DataFrame
        # ====================================================
        cols = model_data['columns']

        input_df = pd.DataFrame(
            0,
            index=[0],
            columns=cols
        )

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

        one_hot_features = [
            f'stand_{stand}',
            f'prev1_pitch_{prev1_pitch}',
            f'prev2_pitch_{prev2_pitch}'
        ]

        for col_name in one_hot_features:
            if col_name in input_df.columns:
                input_df.at[0, col_name] = 1

        # ====================================================
        # 모델
        # ====================================================
        clf_pitch = model_data['pitch_model']
        zone_models = model_data['zone_models']

        # ====================================================
        # 1. 구종 확률
        # ====================================================
        pitch_probs = clf_pitch.predict_proba(input_df)[0]
        pitch_classes = clf_pitch.classes_

        predictions = [
            {
                'pitch': pitch,
                'prob': round(float(prob) * 100, 1)
            }
            for pitch, prob in zip(
                pitch_classes,
                pitch_probs
            )
        ]

        predictions.sort(
            key=lambda x: x['prob'],
            reverse=True
        )

        # ====================================================
        # 2. P(zone) = Σ P(pitch) × P(zone|pitch)
        # ====================================================
        all_zones = model_data['all_zones']

        zone_map = {
            ZONE_NAMES[z]: 0.0
            for z in all_zones
        }

        zone_probability_raw = {
            z: 0.0
            for z in all_zones
        }

        for pitch_idx, pitch in enumerate(pitch_classes):
            if pitch not in zone_models:
                continue

            pitch_probability = pitch_probs[pitch_idx]

            zone_model = zone_models[pitch]

            pitch_zone_probs = zone_model.predict_proba(
                input_df
            )[0]

            for zone_class, zone_prob in zip(
                zone_model.classes_,
                pitch_zone_probs
            ):
                zone_class = int(zone_class)

                if zone_class in zone_probability_raw:
                    zone_probability_raw[zone_class] += (
                        pitch_probability * zone_prob
                    )

        # ====================================================
        # 확률을 %로 변환
        # ====================================================
        total_zone_probability = sum(
            zone_probability_raw.values()
        )

        if total_zone_probability > 0:
            for zone in zone_probability_raw:
                zone_probability_raw[zone] /= total_zone_probability

        for zone in all_zones:
            zone_map[ZONE_NAMES[zone]] = round(
                zone_probability_raw[zone] * 100,
                1
            )

        # ====================================================
        # 결과
        # ====================================================
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'zones': zone_map,
            'accuracy': model_data.get('accuracy', 0.0),
            'top2_accuracy': model_data.get('top2_accuracy', 0.0),
            'zone_accuracy': model_data.get('zone_accuracy', 0.0),
            'pitch_macro_f1': model_data.get('pitch_macro_f1', 0.0),
            'zone_macro_f1': model_data.get('zone_macro_f1', 0.0),
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
