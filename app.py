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

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

MAIN_PITCHES = ['FF', 'FS', 'ST', 'CH', 'CU', 'SI', 'SL']
SEQ_LEN = 5

# 8분할 존(Strike Zone 4개 + Chase Zone 4개) 분류 함수
def assign_zone_8(px, pz):
    in_x = (-0.83 <= px <= 0.83)
    in_z = (1.5 <= pz <= 3.5)
    
    if in_x and in_z:
        if px <= 0 and pz >= 2.5: return 'SZ_UL'
        elif px > 0 and pz >= 2.5: return 'SZ_UR'
        elif px <= 0 and pz < 2.5: return 'SZ_LL'
        else: return 'SZ_LR'
    else:
        if px <= 0 and pz >= 2.5: return 'CHASE_UL'
        elif px > 0 and pz >= 2.5: return 'CHASE_UR'
        elif px <= 0 and pz < 2.5: return 'CHASE_LL'
        else: return 'CHASE_LR'

def create_sequence_features(group, seq_len=5):
    group = group.sort_values('pitch_number').reset_index(drop=True)
    rows = []

    for i in range(len(group)):
        current = group.iloc[i]
        balls = current['balls']
        strikes = current['strikes']
        pitch_num = current['pitch_number']

        px = current['plate_x'] if pd.notna(current['plate_x']) else 0.0
        pz = current['plate_z'] if pd.notna(current['plate_z']) else 2.5

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
            'target_pitch': current['pitch_type_processed'],
            'target_zone': assign_zone_8(px, pz) # 위치 타겟 추가
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
    model_path = os.path.join(MODEL_DIR, f'pitcher_{pitcher_id}.pkl')
    zone_model_path = os.path.join(MODEL_DIR, f'zone_pitcher_{pitcher_id}.pkl')
    encoder_path = os.path.join(MODEL_DIR, f'encoder_{pitcher_id}.pkl')
    zone_encoder_path = os.path.join(MODEL_DIR, f'zone_encoder_{pitcher_id}.pkl')
    cols_path = os.path.join(MODEL_DIR, f'cols_{pitcher_id}.pkl')

    if all(os.path.exists(p) for p in [model_path, zone_model_path, encoder_path, zone_encoder_path, cols_path]):
        print(f"[CACHE] {pitcher_id}번 모델을 기존 파일에서 로드합니다.")
        return joblib.load(model_path), joblib.load(zone_model_path), joblib.load(encoder_path), joblib.load(zone_encoder_path), joblib.load(cols_path)

    print(f"[TRAIN] {pitcher_id}번 선수 데이터를 수집하고 학습을 시작합니다...")
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

    seq_list = []
    for _, group in df.groupby(['game_pk', 'at_bat_number']):
        seq_list.append(create_sequence_features(group, seq_len=SEQ_LEN))
    
    sequence_df = pd.concat(seq_list, ignore_index=True)
    sequence_df = sequence_df[sequence_df['target_pitch'] != 'OTHER'].reset_index(drop=True)

    # 구종 인코더
    label_encoder = LabelEncoder()
    sequence_df['target'] = label_encoder.fit_transform(sequence_df['target_pitch'])

    # 존 위치 인코더
    zone_encoder = LabelEncoder()
    sequence_df['target_zone_encoded'] = zone_encoder.fit_transform(sequence_df['target_zone'])

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
    y_pitch = sequence_df['target'].copy()
    y_zone = sequence_df['target_zone_encoded'].copy()

    categorical_cols = ['stand', 'pitch_combo_1_2']
    for i in range(1, SEQ_LEN + 1):
        categorical_cols += [f'prev{i}_pitch_type', f'prev{i}_description']

    for col in categorical_cols:
        X[col] = X[col].astype('category')

    # 구종 모델
    model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42, objective='multiclass', verbose=-1)
    model.fit(X, y_pitch, categorical_feature=categorical_cols)

    # 위치 존 모델
    zone_model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42, objective='multiclass', verbose=-1)
    zone_model.fit(X, y_zone, categorical_feature=categorical_cols)

    joblib.dump(model, model_path)
    joblib.dump(zone_model, zone_model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(zone_encoder, zone_encoder_path)
    joblib.dump(feature_cols, cols_path)

    return model, zone_model, label_encoder, zone_encoder, feature_cols


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        pitcher_id = int(data.get('pitcher_id', 694973))
        
        model, zone_model, label_encoder, zone_encoder, feature_cols = get_or_train_model(pitcher_id)

        balls = int(data.get('balls', 0))
        strikes = int(data.get('strikes', 0))
        stand = data.get('stand', 'R')
        cum_pitch_count = int(data.get('cum_pitch_count', 1))
        pitch_number = int(data.get('pitch_number', balls + strikes + 1))
        
        is_first_pitch = 1 if (pitch_number == 1 or (balls == 0 and strikes == 0)) else 0

        if is_first_pitch:
            input_dict = {
                'balls': 0, 'strikes': 0, 'outs_when_up': int(data.get('outs_when_up', 0)),
                'inning': int(data.get('inning', 1)), 'stand': stand, 'pitch_number': 1,
                'cum_pitch_count': cum_pitch_count, 'runner_on_1b': int(data.get('runner_on_1b', 0)),
                'runner_on_2b': int(data.get('runner_on_2b', 0)), 'runner_on_3b': int(data.get('runner_on_3b', 0)),
                'is_first_pitch': 1, 'is_pitcher_count': 0, 'is_hitter_count': 0, 'is_2strikes': 0, 'is_full_count': 0,
                'same_pitch_streak': 0, 'at_bat_unique_pitches': 0, 'prev1_is_swing': 0, 'prev1_is_miss': 0,
                'delta_x_prev1_prev2': 0.0, 'delta_z_prev1_prev2': 0.0, 'pitch_combo_1_2': 'NONE_NONE',
                'prev1_pitch_type': 'NONE', 'prev1_description': 'NONE', 'prev1_plate_x': 0.0, 'prev1_plate_z': 2.5,
                'prev2_pitch_type': 'NONE', 'prev2_description': 'NONE', 'prev2_plate_x': 0.0, 'prev2_plate_z': 2.5,
                'prev3_pitch_type': 'NONE', 'prev3_description': 'NONE', 'prev3_plate_x': 0.0, 'prev3_plate_z': 2.5,
                'prev4_pitch_type': 'NONE', 'prev4_description': 'NONE', 'prev4_plate_x': 0.0, 'prev4_plate_z': 2.5,
                'prev5_pitch_type': 'NONE', 'prev5_description': 'NONE', 'prev5_plate_x': 0.0, 'prev5_plate_z': 2.5
            }
        else:
            prev1_pitch = data.get('prev1_pitch', 'FF')
            input_dict = {
                'balls': balls, 'strikes': strikes, 'outs_when_up': int(data.get('outs_when_up', 0)),
                'inning': int(data.get('inning', 1)), 'stand': stand, 'pitch_number': pitch_number,
                'cum_pitch_count': cum_pitch_count, 'runner_on_1b': int(data.get('runner_on_1b', 0)),
                'runner_on_2b': int(data.get('runner_on_2b', 0)), 'runner_on_3b': int(data.get('runner_on_3b', 0)),
                'is_first_pitch': 0, 'is_pitcher_count': 1 if (strikes > balls or strikes == 2) else 0,
                'is_hitter_count': 1 if balls > strikes else 0, 'is_2strikes': 1 if strikes == 2 else 0,
                'is_full_count': 1 if (balls == 3 and strikes == 2) else 0,
                'same_pitch_streak': int(data.get('same_pitch_streak', 1)),
                'at_bat_unique_pitches': int(data.get('at_bat_unique_pitches', 1)),
                'prev1_is_swing': int(data.get('prev1_is_swing', 0)), 'prev1_is_miss': int(data.get('prev1_is_miss', 0)),
                'delta_x_prev1_prev2': 0.0, 'delta_z_prev1_prev2': 0.0, 'pitch_combo_1_2': f"{prev1_pitch}_NONE",
                'prev1_pitch_type': prev1_pitch, 'prev1_description': 'foul', 'prev1_plate_x': 0.0, 'prev1_plate_z': 2.5,
                'prev2_pitch_type': 'NONE', 'prev2_description': 'NONE', 'prev2_plate_x': 0.0, 'prev2_plate_z': 2.5,
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

        # 구종 확률 예측
        proba = model.predict_proba(input_df)[0]
        classes = label_encoder.classes_
        result = [{'pitch': cls, 'prob': round(float(p) * 100, 2)} for cls, p in zip(classes, proba)]
        result = sorted(result, key=lambda x: x['prob'], reverse=True)

        # 위치 존 확률 예측
        zone_proba = zone_model.predict_proba(input_df)[0]
        zone_classes = zone_encoder.classes_
        zone_result = {cls: round(float(p) * 100, 2) for cls, p in zip(zone_classes, zone_proba)}

        return jsonify({"status": "success", "predictions": result, "zones": zone_result})
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
