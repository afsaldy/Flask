from app_reload import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import threading
import requests

app = Flask(__name__)


# Deklarasi endpoint Laravel website
# LARAVEL_KLASIFIKASI_ENDPOINT = "https://weather.kemala-smart-v2.com/api/prediksi-klasifikasi-1"
# LARAVEL_KLASIFIKASI_UDARA_ENDPOINT = "https://weather.kemala-smart-v2.com/api/klasifikasi-udara-1"
# LARAVEL_KALIBRASI_KUALITAS_UDARA_ENDPOINT = "https://weather.kemala-smart-v2.com/api/kalibrasi-kualitas-udara-1"

# Deklarasi endpoint Laravel local
LARAVEL_KLASIFIKASI_ENDPOINT = "http://192.168.1.100:8000/api/prediksi-klasifikasi-1"
LARAVEL_KLASIFIKASI_UDARA_ENDPOINT = "http://192.168.1.100:8000/api/klasifikasi-udara-1"
LARAVEL_KALIBRASI_KUALITAS_UDARA_ENDPOINT = "http://192.168.1.100:8000/api/kalibrasi-kualitas-udara-1"

# Kalibrasi Kualitas Udara
MODEL_KALIBRASI_KUALITAS_UDARA_PATH = 'Model_Kalibrasi\kalibrasi_kualitas_udara_1\model_kalibrasi_kualitas_udara_1.keras'
SCALER_X_KUALITAS_UDARA_PATH = 'Model_Kalibrasi\kalibrasi_kualitas_udara_1\scaler_X_kualitas_udara_1.pkl'
SCALER_Y_KUALITAS_UDARA_PATH = 'Model_Kalibrasi\kalibrasi_kualitas_udara_1\scaler_y_kualitas_udara_1.pkl'

# Prediksi 24 jam & Klasifikasi
ROLLING_CSV_PATH = 'Model_Prediksi\prediksi_1/rolling_window_1.csv'
MODEL_PREDIKSI_PATH = 'Model_Prediksi\prediksi_1\model_prediksi_1.keras'
SCALER_PREDIKSI_PATH = 'Model_Prediksi\prediksi_1\scaler_prediksi_1.pkl'
MODEL_KLASIFIKASI_PATH = 'Model_Klasifikasi\klasifikasi_1\model_klasifikasi_1.pkl'
SCALER_KLASIFIKASI_PATH = 'Model_Klasifikasi\klasifikasi_1\scaler_klasifikasi_1.pkl'
ENCODER_KLASIFIKASI_PATH = 'Model_Klasifikasi\klasifikasi_1\label_encoder_klasifikasi_1.pkl'
WINDOW_SIZE = 168
SEQ_LENGTH = 24

# validasi input
def validate_input(data, required_fields):
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Field berikut wajib ada: {', '.join(missing)}"
    return True, ""

# Fungsi kalibrasi kualitas udara
def load_model_and_scalers_kalibrasi():
    model = load_model(MODEL_KALIBRASI_KUALITAS_UDARA_PATH)
    with open(SCALER_X_KUALITAS_UDARA_PATH, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(SCALER_Y_KUALITAS_UDARA_PATH, 'rb') as f:
        scaler_y = pickle.load(f)
    return model, scaler_X, scaler_y

@app.route('/kalibrasi_kualitas_udara_1', methods=['POST'])
def kalibrasi_kualitas_udara():
    try:
        data = request.get_json()
        required_fields = [
            'Suhu (C)', 'Kelembaban (%)', 'Tekanan Udara (hPa)', 'Kecepatan Angin (m/s)',
            'PM 2.5 (ug/m3)', 'CO (ADC)', 'CH4 (ADC)', 'OZON (ADC)'
        ]
        valid, msg = validate_input(data, required_fields)
        if not valid:
            return jsonify({"status": "error", "message": msg}), 400
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        model, scaler_X, scaler_y = load_model_and_scalers_kalibrasi()
        features = required_fields
        X_input = pd.DataFrame([[data[f] for f in features]], columns=features)
        # DEBUG: Print kolom dan isi DataFrame sebelum scaling
        print("DEBUG - X_input.columns:", X_input.columns.tolist())
        print("DEBUG - X_input values:", X_input.values)
        X_input_scaled = scaler_X.transform(X_input)
        y_pred_scaled = model.predict(X_input_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        result = {
            "pm25_bmkg": float(y_pred[0][0]),
            "co_bmkg": float(y_pred[0][1]),
            "ch4_bmkg": float(y_pred[0][2]),
            "ozon_bmkg": float(y_pred[0][3])
        }

        # Kirim hasil ke endpoint Laravel kalibrasi kualitas udara
        try:
            laravel_url = LARAVEL_KALIBRASI_KUALITAS_UDARA_ENDPOINT
            resp = requests.post(laravel_url, json=result, timeout=5)
            resp.raise_for_status()
            laravel_response = resp.json()
        except Exception as e:
            laravel_response = {"error": str(e)}

        return jsonify({"status": "success", "prediction": result, "laravel_response": laravel_response})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Fungsi Prediksi 24 Jam + Klasifikasi ---
def update_rolling_window(df_new):
    if os.path.exists(ROLLING_CSV_PATH):
        df_hist = pd.read_csv(ROLLING_CSV_PATH)
        if not all(col in df_new.columns for col in df_hist.columns):
            raise ValueError(f"Kolom data baru tidak cocok dengan header rolling window: {df_hist.columns.tolist()}")
        df_new = df_new[df_hist.columns]
        df_hist = pd.concat([df_hist, df_new], ignore_index=True)
        df_hist = df_hist.drop_duplicates(subset=['Timestamp'], keep='last')
    else:
        df_hist = df_new.copy()
    if len(df_hist) > WINDOW_SIZE:
        df_hist = df_hist.tail(WINDOW_SIZE)
    df_hist.to_csv(ROLLING_CSV_PATH, index=False)
    return df_hist

def retrain_model():
    print("Mulai retrain model...")  # Notifikasi awal retrain
    df = pd.read_csv(ROLLING_CSV_PATH)
    if len(df) < SEQ_LENGTH + 1:
        return False
    if os.path.exists(SCALER_PREDIKSI_PATH):
        with open(SCALER_PREDIKSI_PATH, 'rb') as f:
            scaler = pickle.load(f)
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.iloc[:, 1:])
    X, y = [], []
    for i in range(len(data_scaled) - SEQ_LENGTH):
        X.append(data_scaled[i:i+SEQ_LENGTH])
        y.append(data_scaled[i+SEQ_LENGTH])
    X, y = np.array(X), np.array(y)
    if os.path.exists(MODEL_PREDIKSI_PATH):
        model = load_model(MODEL_PREDIKSI_PATH, compile=False)
        model.compile(optimizer='adam', loss='mse')
    else:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dropout, Dense
        model = Sequential([
            LSTM(64, input_shape=(SEQ_LENGTH, X.shape[2]), return_sequences=False),
            Dropout(0.2),
            Dense(X.shape[2])
        ])
        model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)
    model.save(MODEL_PREDIKSI_PATH)
    with open(SCALER_PREDIKSI_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print("Retrain model selesai.")  # Notifikasi selesai retrain
    return True

def predict_24h(start_timestamp=None):
    df = pd.read_csv(ROLLING_CSV_PATH)
    if len(df) < SEQ_LENGTH:
        return []
    with open(SCALER_PREDIKSI_PATH, 'rb') as f:
        scaler = pickle.load(f)
    model = load_model(MODEL_PREDIKSI_PATH, compile=False)
    data_scaled = scaler.transform(df.iloc[:, 1:])
    last_seq = data_scaled[-SEQ_LENGTH:]
    preds = []
    input_seq = last_seq.copy()
    for _ in range(24):
        pred = model.predict(input_seq[np.newaxis, :, :], verbose=0)[0]
        preds.append(pred)
        input_seq = np.vstack([input_seq[1:], pred])
    preds = scaler.inverse_transform(preds)
    pred_list = []
    # Gunakan start_timestamp jika diberikan, jika tidak pakai timestamp terakhir rolling window
    if start_timestamp is not None:
        last_time = pd.to_datetime(start_timestamp)
    else:
        last_time = pd.to_datetime(df.iloc[-1, 0])
    for i, row in enumerate(preds):
        pred_time = last_time + pd.Timedelta(hours=i+1)
        pred_dict = {"Timestamp": str(pred_time)}
        for j, col in enumerate(df.columns[1:]):
            pred_dict[col] = float(row[j])
        pred_list.append(pred_dict)
    return pred_list

@app.route('/predik_klasifikasi_1', methods=['POST'])
def predik_klasifikasi():
    try:
        data = request.get_json()
        required_fields = ['Timestamp', 'Suhu (C)', 'Kelembaban (%)', 'Tekanan Udara (hPa)', 'Kecepatan Angin (m/s)']
        valid, msg = validate_input(data, required_fields)
        if not valid:
            return jsonify({"status": "error", "message": msg}), 400
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        # Tambahkan data input ke rolling window sebelum prediksi
        df_new = pd.DataFrame([data])
        update_rolling_window(df_new)
        # Gunakan timestamp input user sebagai acuan prediksi
        preds = predict_24h(start_timestamp=data['Timestamp'])
        with open(MODEL_KLASIFIKASI_PATH, 'rb') as f:
            model_klasifikasi = pickle.load(f)
        with open(SCALER_KLASIFIKASI_PATH, 'rb') as f:
            scaler_klasifikasi = pickle.load(f)
        with open(ENCODER_KLASIFIKASI_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        fitur_klasifikasi = ['Suhu (C)', 'Kelembaban (%)', 'Tekanan Udara (hPa)', 'Kecepatan Angin (m/s)']
        X_pred = [[p[f] for f in fitur_klasifikasi] for p in preds]
        X_pred_scaled = scaler_klasifikasi.transform(X_pred)
        kategori_pred = model_klasifikasi.predict(X_pred_scaled)
        kategori_pred_label = label_encoder.inverse_transform(kategori_pred)
        for i, p in enumerate(preds):
            p['kategori_hujan'] = kategori_pred_label[i]
        response = {
            "message": "Prediction generated. Data will be updated and model retrained in background.",
            "prediksi_24_jam": preds
        }
        # Kirim seluruh hasil prediksi 24 jam ke Laravel dalam satu array (batch)
        payload_batch = []
        for p in preds:
            payload_batch.append({
                "timestamp": p["Timestamp"],
                "suhu": p["Suhu (C)"],
                "kelembaban": p["Kelembaban (%)"],
                "tekananudara": p["Tekanan Udara (hPa)"],
                "kecepatanangin": p["Kecepatan Angin (m/s)"],
                "kategori_cuaca": p["kategori_hujan"]
            })
        try:
            resp = requests.post(LARAVEL_KLASIFIKASI_ENDPOINT, json=payload_batch, timeout=10)
            resp.raise_for_status()
            laravel_response = resp.json()
        except Exception as e:
            laravel_response = {"error": str(e)}
        # Update rolling window dan retrain model di latar belakang
        def update_and_retrain():
            retrain_model()
        threading.Thread(target=update_and_retrain).start()
        return jsonify({"message": response["message"], "prediksi_24_jam": preds, "laravel_response": laravel_response})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# fungsi prediksi_klasifikasi (nilai realtime sensor)
@app.route('/klasifikasi_udara_1', methods=['POST'])
def klasifikasi_udara():
    try:
        data = request.get_json()
        required_fields = ['Suhu (C)', 'Kelembaban (%)', 'Tekanan Udara (hPa)', 'Kecepatan Angin (m/s)']
        valid, msg = validate_input(data, required_fields)
        if not valid:
            return jsonify({"status": "error", "message": msg}), 400
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        # Load model, scaler, dan encoder klasifikasi
        with open(MODEL_KLASIFIKASI_PATH, 'rb') as f:
            model_klasifikasi = pickle.load(f)
        with open(SCALER_KLASIFIKASI_PATH, 'rb') as f:
            scaler_klasifikasi = pickle.load(f)
        with open(ENCODER_KLASIFIKASI_PATH, 'rb') as f:
            label_encoder = pickle.load(f)

        fitur_klasifikasi = ['Suhu (C)', 'Kelembaban (%)', 'Tekanan Udara (hPa)', 'Kecepatan Angin (m/s)']
        X_input = [[data[f] for f in fitur_klasifikasi]]
        X_input_scaled = scaler_klasifikasi.transform(X_input)
        kategori_pred = model_klasifikasi.predict(X_input_scaled)
        kategori_pred_label = label_encoder.inverse_transform(kategori_pred)[0]

        # Gabungkan hasil prediksi dengan data input
        result = {
            "suhu": data["Suhu (C)"],
            "kelembaban": data["Kelembaban (%)"],
            "tekananudara": data["Tekanan Udara (hPa)"],
            "kecepatanangin": data["Kecepatan Angin (m/s)"],
            "kategori_cuaca": kategori_pred_label
        }

        # Kirim hasil ke endpoint Laravel
        try:
            laravel_url = LARAVEL_KLASIFIKASI_UDARA_ENDPOINT
            resp = requests.post(laravel_url, json=result, timeout=5)
            resp.raise_for_status()
            laravel_response = resp.json()
        except Exception as e:
            laravel_response = {"error": str(e)}

        return jsonify({"status": "success", "result": result, "laravel_response": laravel_response})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)