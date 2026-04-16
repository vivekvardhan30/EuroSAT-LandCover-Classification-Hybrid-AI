from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os, uuid, json
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")   # ✅ Fix GUI thread issue
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import joblib
from skimage.feature import graycomatrix, graycoprops

# -------------------------------
# Helper: find last conv layer name
# -------------------------------
def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            return layer.name
    return model.layers[-1].name

# -------------------------------
# Grad-CAM Heatmap
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        top_index = tf.argmax(predictions[0])
        loss = predictions[:, top_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    return heatmap

def save_and_return_gradcam(img_path, model, save_path, alpha=0.4, target_size=(64, 64)):
    pil_img = load_img(img_path, target_size=target_size)
    img_arr = img_to_array(pil_img) / 255.0
    input_arr = np.expand_dims(img_arr, axis=0)

    last_conv = find_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(input_arr, model, last_conv)
    heatmap = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_uint8 = np.uint8(img_arr * 255)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    superimposed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(save_path, superimposed)

    return save_path

# -------------------------------
# Feature Extraction for RF Model
# -------------------------------
def extract_features(image_path, is_multiband=False):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if is_multiband and img is not None and len(img.shape) == 3 and img.shape[2] >= 13:
        bands = [img[:,:,i] for i in range(13)]
        means = [np.mean(b) for b in bands]
        stds = [np.std(b) for b in bands]
        nir, red = bands[7], bands[3]
        ndvi = np.mean((nir - red) / (nir + red + 1e-5))
        return means + stds + [ndvi]

    elif img is not None:
        img_rgb = cv2.resize(img, (64,64))
        means, stds = [], []
        for i in range(3):
            means.append(np.mean(img_rgb[:,:,i]))
            stds.append(np.std(img_rgb[:,:,i]))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=[5], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0,0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
        return means + stds + [contrast, homogeneity]
    return None

# -------------------------------
# Initialize Flask App
# -------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -------------------------------
# Load Models
# -------------------------------
custom_model = load_model("saved_models/eurosat_custom_cnn.h5")
vgg_model = load_model("saved_models/eurosat_vgg16.h5")
rf_model = joblib.load("saved_models/eurosat_randomforest.pkl")

with open("saved_models/class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {int(v): k for k, v in class_indices.items()}

# -------------------------------
# Prediction Functions
# -------------------------------
def predict_image(model, img_path):
    img = load_img(img_path, target_size=(64,64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    class_index = np.argmax(preds)
    confidence = preds[class_index]
    return index_to_class[class_index], confidence, preds

def predict_rf(img_path):
    f = extract_features(img_path, is_multiband=False)
    if f is None:
        return "Error", 0.0, []
    preds = rf_model.predict_proba([f])[0]
    idx = np.argmax(preds)
    return index_to_class[idx], preds[idx], preds

# -------------------------------
# Probability Distribution Plot
# -------------------------------
def save_probability_chart(probabilities, class_names, save_path):
    plt.figure(figsize=(8, 4))
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, probabilities, color="skyblue")
    plt.yticks(y_pos, class_names)
    plt.xlabel("Probability")
    plt.title("Prediction Confidence Distribution")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        pred_custom, conf_custom, probs_custom = predict_image(custom_model, filepath)
        pred_vgg, conf_vgg, probs_vgg = predict_image(vgg_model, filepath)
        pred_rf, conf_rf, probs_rf = predict_rf(filepath)

        prob_chart_path = os.path.join(STATIC_FOLDER, f"prob_chart_{filename}.png")
        save_probability_chart(probs_custom, list(index_to_class.values()), prob_chart_path)

        return render_template("index.html",
            file_path=url_for('uploaded_file', filename=filename),
            custom_pred=pred_custom,
            custom_conf=float(round(conf_custom*100, 2)),
            vgg_pred=pred_vgg,
            vgg_conf=float(round(conf_vgg*100, 2)),
            rf_pred=pred_rf,
            rf_conf=float(round(conf_rf*100, 2)),
            probs_custom=probs_custom.tolist(),
            probs_vgg=probs_vgg.tolist(),
            probs_rf=probs_rf.tolist(),
            class_names=list(index_to_class.values())
        )


    return render_template("index.html")



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
