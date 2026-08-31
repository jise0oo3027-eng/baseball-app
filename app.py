import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from pybaseball import statcast_pitcher
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# 모델 저장 디렉토리 생성
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

MAIN_PITCHES = ['FF', 'FS', 'ST', 'CH', 'CU', 'SI', 'SL']
SEQ_LEN = 5

def create_sequence_features(group, seq_len=5):
    group = group.sort_values('pitch_number').reset_index(drop=True)
    rows = []

    for i in range(len(group)):
        current = group.iloc[i]
        balls = current['balls']
        strikes = current['strikes']
        pitch_num = current['pitch_number']

        row = {
            'balls': balls,
            'strikes': strikes,
            'outs_when_up': current['outs_when_up'],
            'inning': current['inning'],
            'stand': current['stand'],
            'pitch_number': pitch_num,
            'cum_pitch_count': current['cum_pitch_count'],
            'runner_on_1b': current['runner_on_1b'],
            'runner_on_2b': current['runner_on_2b'],
            'runner_on_3b': current['runner_on_3b'],
            'target_pitch': current['pitch_type_processed']
        }

        row['is_first_pitch'] = 1 if pitch_num == 1 else 0
        row['is_pitcher_count'] = 1 if (strikes > balls or strikes == 2) else 0
        row['is_hitter_count'] = 1 if (balls > strikes) else 0
        row['is_2strikes'] = 1 if (strikes == 2) else 0
        row['is_full_count'] = 1 if (balls == 3 and strikes == 2) else 0

        start_idx = max(0, i - seq_len)
        history = group.iloc[start_idx:i]

        hist_pitches = history['pitch_type_processed'].tolist()[::-1]
        hist_descs = history['description'].tolist()[::-1]
        hist_xs = history['plate_x'].tolist()[::-1]
        hist_zs = history['plate_z'].tolist()[::-1]

        same_streak = 0
        if len(hist_pitches) > 0:
            first_p = hist_pitches[0]
            for p in hist_pitches:
                if p == first_p:
                    same_streak += 1
                else:
                    break
        row['same_pitch_streak'] = same_streak
        row['at_bat_unique_pitches'] = len(set(hist_pitches))

        row['prev1_is_swing'] = 1 if len(hist_descs) > 0 and 'swing' in str(hist_descs[0]) else 0
        row['prev1_is_miss'] = 1 if len(hist_descs) > 0 and 'swinging_strike' in str(hist_descs[0]) else 0

        if len(hist_xs) >= 2:
            row['delta_x_prev1_prev2'] = hist_xs[0] - hist_xs[1]
            row['delta_z_prev1_prev2'] = hist_zs[0] - hist_zs[1]
        else:
            row['delta_x_prev1_prev2'] = 0.0
            row['delta_z_prev1_prev2'] = 0.0

        if len(hist_pitches) >= 2:
            row['pitch_combo_1_2'] = f"{hist_pitches[0]}_{hist_pitches[1]}"
        elif len(hist_pitches) == 1:
            row['pitch_combo_1_2'] = f"{hist_pitches[0]}_NONE"
        else:
            row['pitch_combo_1_2'] = "NONE_NONE"

        for j in range(1, seq_len + 1):
            idx = j - 1
            if idx < len(hist_pitches):
                row[f'prev{j}_pitch_type'] = hist_pitches[idx]
                row[f'prev{j}_description'] = hist_descs[idx]
                row[f'prev{j}_plate_x'] = hist_xs[idx]
                row[f'prev{j}_plate_z'] = hist_zs[idx]
            else:
                row[f'prev{j}_pitch_type'] = 'NONE'
                row[f'prev{j}_description'] = 'NONE'
                row[f'prev{j}_plate_x'] = 0.0
                row[f'prev{j}_plate_z'] = 2.5

        rows.append(row)

    return pd.DataFrame(rows)


def get_or_train_model(pitcher_id):
    """모델 캐싱 및 온디맨드 학습 처리"""
    model_path = os.path.join(MODEL_DIR, f'pitcher_{pitcher_id}.pkl')
    encoder_path = os.path.join(MODEL_DIR, f'encoder_{pitcher_id}.pkl')
    cols_path = os.path.join(MODEL_DIR, f'cols_{pitcher_id}.pkl')

    # 캐시 존재 시 로드
    if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(cols_path):
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        feature_cols = joblib.load(cols_path)
        return model, label_encoder, feature_cols

    # 신규 수집 및 학습
    df = statcast_pitcher('2024-03-01', '2026-10-31', pitcher_id)
    df = df[df['game_type'] == 'R'].copy()
    df = df.sort_values(by=['game_date', 'game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)
    df['cum_pitch_count'] = df.groupby(['game_pk']).cumcount() + 1

    df['runner_on_1b'] = df['on_1b'].notna().astype(int)
    df['runner_on_2b'] = df['on_2b'].notna().astype(int)
    df['runner_on_3b'] = df['on_3b'].notna().astype(int)
    df['stand'] = df['stand'].fillna('UNKNOWN')
    df['description'] = df['description'].fillna('unknown')
    df['plate_x'] = df['plate_x'].fillna(df['plate_x'].median())
    df['plate_z'] = df['plate_z'].fillna(df['plate_z'].median())

    df['pitch_type_processed'] = df['pitch_type'].apply(lambda x: x if x in MAIN_PITCHES else 'OTHER')

    sequence_df = (
        df.groupby(['game_pk', 'at_bat_number'], group_keys=False)
        .apply(create_sequence_features, seq_len=SEQ_LEN, include_groups=False)
        .reset_index(drop=True)
    )

    sequence_df = sequence_df[sequence_df['target_pitch'] != 'OTHER'].reset_index(drop=True)

    label_encoder = LabelEncoder()
    sequence_df['target'] = label_encoder.fit_transform(sequence_df['target_pitch'])

    feature_cols = [
        'balls', 'strikes', 'outs_when_up', 'inning', 'stand', 'pitch_number',
        'cum_pitch_count', 'runner_on_1b', 'runner_on_2b', 'runner_on_3b',
        'is_first_pitch', 'is_pitcher_count', 'is_hitter_count', 'is_2strikes',
        'is_full_count', 'same_pitch_streak', 'at_bat_unique_pitches',
        'prev1_is_swing', 'prev1_is_miss', 'delta_x_prev1_prev2', 'delta_z_prev1_prev2',
        'pitch_combo_1_2'
    ]
    for i in range(1, SEQ_LEN + 1):
        feature_cols += [
            f'prev{i}_pitch_type', f'prev{i}_description',
            f'prev{i}_plate_x', f'prev{i}_plate_z'
        ]

    X = sequence_df[feature_cols].copy()
    y = sequence_df['target'].copy()

    categorical_cols = ['stand', 'pitch_combo_1_2']
    for i in range(1, SEQ_LEN + 1):
        categorical_cols += [f'prev{i}_pitch_type', f'prev{i}_description']

    for col in categorical_cols:
        X[col] = X[col].astype('category')

    # 클래스 균형 기반 동적 가중치 자동 산출
    class_counts = y.value_counts().to_dict()
    total_samples = len(y)
    n_classes = len(class_counts)
    dynamic_cw = {cls: total_samples / (n_classes * count) for cls, count in class_counts.items()}

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        class_weight=dynamic_cw,
        random_state=42,
        objective='multiclass',
        verbose=-1
    )
    model.fit(X, y, categorical_feature=categorical_cols)

    # 캐시 파일 저장
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(feature_cols, cols_path)

    return model, label_encoder, feature_cols


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pitcher_id = int(data.get('pitcher_id', 694973))
    
    # 모델 확보 (학습 또는 파일 불러오기)
    model, label_encoder, feature_cols = get_or_train_model(pitcher_id)

    balls = int(data.get('balls', 0))
    strikes = int(data.get('strikes', 0))
    stand = data.get('stand', 'R')
    cum_pitch_count = int(data.get('cum_pitch_count', 50))
    prev1_pitch = data.get('prev1_pitch', 'FF')

    # 단일 시뮬레이션용 입력 데이터 생성
    input_dict = {
        'balls': balls,
        'strikes': strikes,
        'outs_when_up': 1,
        'inning': 3,
        'stand': stand,
        'pitch_number': 3,
        'cum_pitch_count': cum_pitch_count,
        'runner_on_1b': 0, 'runner_on_2b': 0, 'runner_on_3b': 0,
        'is_first_pitch': 0,
        'is_pitcher_count': 1 if strikes > balls or strikes == 2 else 0,
        'is_hitter_count': 1 if balls > strikes else 0,
        'is_2strikes': 1 if strikes == 2 else 0,
        'is_full_count': 1 if (balls == 3 and strikes == 2) else 0,
        'same_pitch_streak': 1,
        'at_bat_unique_pitches': 2,
        'prev1_is_swing': 1,
        'prev1_is_miss': 0,
        'delta_x_prev1_prev2': 0.1,
        'delta_z_prev1_prev2': -0.2,
        'pitch_combo_1_2': f"{prev1_pitch}_FF",
        'prev1_pitch_type': prev1_pitch, 'prev1_description': 'foul', 'prev1_plate_x': 0.1, 'prev1_plate_z': 2.3,
        'prev2_pitch_type': 'FF', 'prev2_description': 'ball', 'prev2_plate_x': 0.0, 'prev2_plate_z': 2.5,
        'prev3_pitch_type': 'NONE', 'prev3_description': 'NONE', 'prev3_plate_x': 0.0, 'prev3_plate_z': 2.5,
        'prev4_pitch_type': 'NONE', 'prev4_description': 'NONE', 'prev4_plate_x': 0.0, 'prev4_plate_z': 2.5,
        'prev5_pitch_type': 'NONE', 'prev5_description': 'NONE', 'prev5_plate_x': 0.0, 'prev5_plate_z': 2.5
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]

    categorical_cols = ['stand', 'pitch_combo_1_2']
    for i in range(1, SEQ_LEN + 1):
        categorical_cols += [f'prev{i}_pitch_type', f'prev{i}_description']

    for col in categorical_cols:
        input_df[col] = input_df[col].astype('category')

    proba = model.predict_proba(input_df)[0]
    classes = label_encoder.classes_

    result = []
    for cls, p in zip(classes, proba):
        result.append({'pitch': cls, 'prob': round(float(p) * 100, 2)})

    result = sorted(result, key=lambda x: x['prob'], reverse=True)

    return jsonify({"status": "success", "predictions": result})


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
