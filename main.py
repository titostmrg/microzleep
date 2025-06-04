from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
import json
import mediapipe as mp
import os

app = FastAPI()

# Inisialisasi MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Indeks landmark untuk mata dari MediaPipe
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Fungsi untuk menghitung EAR (Eye Aspect Ratio)
def calculate_ear(landmarks, left_indices, right_indices):
    def ear_single_eye(eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)

    left_eye = [landmarks[i] for i in left_indices]
    right_eye = [landmarks[i] for i in right_indices]

    return (ear_single_eye(left_eye) + ear_single_eye(right_eye)) / 2.0

# Load model TFLite
interpreter = None
try:
    interpreter = tf.lite.Interpreter(model_path="model_microsleep2.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model TFLite berhasil dimuat.")

    # --- TAMBAHKAN DEBUG LOG INI ---
    print("\n--- Model Input Details ---")
    for i, detail in enumerate(input_details):
        print(f"Input {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Dtype: {detail['dtype']}")
    print("---------------------------\n")
    # -----------------------------

except Exception as e:
    print(f"Error memuat model TFLite: {e}")

# Load scaler parameters
scaler_mean = None
scaler_std = None
try:
    with open("ear_scaler_params.json", "r") as f:
        scaler_params = json.load(f)
        scaler_mean = np.array(scaler_params["mean"])
        scaler_std = np.array(scaler_params["std"])
    print("Parameter scaler EAR berhasil dimuat.")
except Exception as e:
    print(f"Error memuat parameter scaler: {e}")

# Preprocessing gambar untuk model
def preprocess_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Tidak dapat menguraikan gambar")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_array = img_resized / 255.0
    return np.expand_dims(img_array, axis=0), img # Mengembalikan img asli juga untuk ekstraksi EAR

# Endpoint untuk deteksi microsleep
@app.post("/predict-microsleep/")
async def predict_microsleep(file: UploadFile = File(...)):
    if interpreter is None or scaler_mean is None or scaler_std is None:
        raise HTTPException(status_code=500, detail="Model atau scaler belum dimuat dengan benar.")

    image_bytes = await file.read()
    img_preprocessed, original_img = preprocess_image(image_bytes)

    # Ekstraksi EAR dari gambar asli (BGR)
    ear_value = None
    results = face_mesh.process(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = original_img.shape
        landmark_coords = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        ear_value = calculate_ear(landmark_coords, LEFT_EYE_IDX, RIGHT_EYE_IDX)

    if ear_value is None:
        raise HTTPException(status_code=400, detail="Wajah tidak terdeteksi atau EAR tidak dapat dihitung.")

    ear_scaled_value = (ear_value - scaler_mean[0]) / scaler_std[0] # Skalakan nilai EAR
    ear_input = np.array([[ear_scaled_value]], dtype=np.float32)

    # --- PERBAIKAN UTAMA DI SINI ---
    # Model Anda memiliki 2 input: gambar (4D) dan EAR (2D).
    # Pesan error menunjukkan 'input 0' mengharapkan 2 dimensi,
    # yang berarti input EAR mungkin adalah input pertama (indeks 0),
    # dan input gambar adalah input kedua (indeks 1).

    # Mari kita asumsikan input_details[0] adalah EAR dan input_details[1] adalah gambar
    # Berdasarkan summary model di notebook, 'img_input' (gambar) adalah input pertama,
    # dan 'ear_input' adalah input kedua. Tapi TFLite bisa mengubah urutan.
    # kita akan cek nama input_details[0]['name'] dan input_details[1]['name']

    # Cek nama input
    input_0_name = input_details[0]['name']
    input_1_name = input_details[1]['name']

    # Tentukan mana yang gambar dan mana yang EAR berdasarkan nama input
    image_input_index = -1
    ear_input_index = -1

    if 'img_input' in input_0_name.lower(): # Perhatikan 'img_input' adalah nama layer Keras Anda
        image_input_index = 0
        ear_input_index = 1
    elif 'ear_input' in input_0_name.lower():
        image_input_index = 1
        ear_input_index = 0
    else:
        # Jika nama tidak dikenali, coba asumsikan urutan standar (gambar dulu, lalu EAR)
        # Atau log lebih banyak untuk debugging
        print(f"Warning: Input names not recognized. Assuming image_input_index=0, ear_input_index=1. Input 0 name: {input_0_name}, Input 1 name: {input_1_name}")
        image_input_index = 0
        ear_input_index = 1


    # Set input tensor sesuai dengan urutan yang benar
    interpreter.set_tensor(input_details[image_input_index]['index'], img_preprocessed.astype(np.float32))
    interpreter.set_tensor(input_details[ear_input_index]['index'], ear_input)
    # ------------------------------------

    # Jalankan inferensi
    interpreter.invoke()

    # Dapatkan hasil prediksi
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "MICROSLEEP" if prediction > 0.5 else "NORMAL"

    return {"prediction": float(prediction), "label": label, "ear_value": float(ear_value)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)