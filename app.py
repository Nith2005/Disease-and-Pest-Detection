import streamlit as st
from PIL import Image
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from pytorch_tabnet.tab_model import TabNetClassifier
import cv2

# -------------------------------
# Load YOLO models (adjust paths)
# -------------------------------
yolo1 = YOLO('E:\\Agrithon\\temp\\best.pt')
yolo2 = YOLO('E:\\Agrithon\\temp\\best1.pt')

# Load TabNet model (adjust path)
tabnet = TabNetClassifier()
tabnet.load_model('E:\\Agrithon\\tabnet_model.zip.zip')

# -------------------------------
# Visual question features (technical names used by TabNet)
# Order matches your CSV: YL_Visual (7) -> YL_Context (4) -> RS_Visual (6) -> RS_Context (5)
# -------------------------------
visual_questions = {
    "Yellow Leaf Disease": [
        "YL_Visual_LeafMidribYellowing",
        "YL_Visual_Leaves3to6Yellowing",
        "YL_Visual_DryingMidrib",
        "YL_Visual_LeavesReddishPink",
        "YL_Visual_SpindleBunchyDrying",
        "YL_Visual_StuntedCane",
        "YL_Visual_BleachedNecrotic"
    ],
    "Ring Spot Disease": [
        "RS_Visual_SmallSpots",
        "RS_Visual_SpotsDarkGreenYellowHalo",
        "RS_Visual_IrregularOutlines",
        "RS_Visual_SpotsMerged",
        "RS_Visual_SpotsOlderLeaves",
        "RS_Visual_BlackDotsInLesions"
    ]
}

# -------------------------------
# Contextual questions: Friendly text -> technical feature name
# Farmer sees the friendly text. Answers are mapped to the technical keys.
# -------------------------------
friendly_contextual_questions = {
    "Yellow Leaf Disease": {
        "Was the weather recently wet or humid?": "YL_Context_HumidWet",   # map to your actual trained feature names
        "Do you see any small black or dark spots on the leaf?": "YL_Context_BlackSpots",
        "Are the lower surface or midribs showing yellowing?": "YL_Context_MidribYellowing",
        "Has there been lots of aphids or sticky residue on leaves?": "YL_Context_AphidsPresent"
    },
    "Ring Spot Disease": {
        "Do you see round/oval rings or spots on the leaves?": "RS_Context_RingLikeSpots",
        "Are the spots darker in the center with a yellow halo?": "RS_Context_DarkCenterYellowHalo",
        "Has the field been wet and warm recently?": "RS_Context_HumidWarmConditions",
        "Are multiple older leaves showing similar spots?": "RS_Context_MultipleOlderLeaves"
    }
}

# -------------------------------
# Build master feature list in exact order as CSV (without YOLO_Predicted flag)
# all_questions length should be 22 according to your training CSV
# -------------------------------
all_questions = []
# YL visual (7)
all_questions += visual_questions["Yellow Leaf Disease"]
# YL contextual (4) — use the technical keys from friendly_contextual_questions
all_questions += list(friendly_contextual_questions["Yellow Leaf Disease"].values())
# RS visual (6)
all_questions += visual_questions["Ring Spot Disease"]
# RS contextual (5)
all_questions += list(friendly_contextual_questions["Ring Spot Disease"].values())

# Sanity check (optional) - you can comment this out
# print(len(all_questions), all_questions)

# -------------------------------
# Helper functions
# -------------------------------
def normalize_disease_name(name):
    norm = name.replace('_', ' ').title()
    if 'Yello' in norm:
        norm = norm.replace('Yello', 'Yellow')
    return norm

def save_annotated_image(result, save_path):
    try:
        annotated_img = result.plot()  # returns BGR numpy array
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_img)
        img_pil.save(save_path)
    except Exception as e:
        # fallback: save original image
        print("Warning saving annotated image:", e)
        if hasattr(result, "orig_img"):
            img = Image.fromarray(result.orig_img)
            img.save(save_path)

def run_yolo_and_annotate(image_path):
    results1 = yolo1.predict(image_path, conf=0.25)
    results2 = yolo2.predict(image_path, conf=0.25)

    detections_info = []
    combined_conf = {}

    def extract_info(results, model_name):
        rows = []
        for r in results:
            try:
                for box in r.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    label = r.names[cls_id]
                    coords = box.xyxy.cpu().numpy()[0].tolist()
                    rows.append({
                        "Class": label,
                        "Confidence": round(conf, 3),
                        "BBox": [round(float(x), 1) for x in coords],
                        "Model": model_name
                    })
            except Exception:
                continue
        return rows

    rows1 = extract_info(results1, "YOLO-1")
    rows2 = extract_info(results2, "YOLO-2")

    for r in rows1 + rows2:
        detections_info.append(r)
        cls = r["Class"]
        conf = r["Confidence"]
        if cls not in combined_conf or conf > combined_conf[cls]:
            combined_conf[cls] = conf

    folder1 = 'temp_annotated1'
    folder2 = 'temp_annotated2'
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

    annotated_img1 = os.path.join(folder1, 'annotated1.jpg')
    annotated_img2 = os.path.join(folder2, 'annotated2.jpg')

    if len(results1) > 0:
        save_annotated_image(results1[0], annotated_img1)
    else:
        Image.open(image_path).save(annotated_img1)

    if len(results2) > 0:
        save_annotated_image(results2[0], annotated_img2)
    else:
        Image.open(image_path).save(annotated_img2)

    detected = set(combined_conf.keys())
    return detected, combined_conf, annotated_img1, annotated_img2, detections_info

def build_feature_vector_from_yolo(disease_key, detected_classes):
    # Build visual feature dictionary (only visual flags for the disease set to 1)
    input_dict = {q: 0 for q in all_questions}
    if disease_key == "Yellow Leaf Disease":
        for vq in visual_questions["Yellow Leaf Disease"]:
            input_dict[vq] = 1 if any("yellow" in d.lower() or "yello" in d.lower() or "yl" in d.lower() for d in detected_classes) else 0
    elif disease_key == "Ring Spot Disease":
        for vq in visual_questions["Ring Spot Disease"]:
            input_dict[vq] = 1 if any("ring" in d.lower() or "spot" in d.lower() or "rs" in d.lower() for d in detected_classes) else 0
    # By default, the contextual keys remain 0 until farmer answers
    return input_dict

def predict_disease_with_answers(disease, farmer_answers_map, yolo_visual_inputs):
    # Start with all zeros in the same order as all_questions
    input_dict = {q: 0 for q in all_questions}
    # Fill visual features detected by YOLO
    input_dict.update(yolo_visual_inputs)
    # Fill contextual answers provided by farmer (technical keys -> 0/1)
    for technical_key, ans in farmer_answers_map.items():
        if technical_key in input_dict:
            input_dict[technical_key] = 1 if ans else 0

    # IMPORTANT: Do NOT prepend any extra flag. Build X matching training shape (n_features == len(all_questions))
    row = [input_dict[q] for q in all_questions]
    X = np.array([row], dtype=np.float32)  # shape (1, 22) expected
    try:
        pred = tabnet.predict(X)
    except Exception as e:
        st.error(f"Error during TabNet prediction: {e}")
        # For debugging, show shapes and keys
        st.write("Feature count expected by model:", "see model's input shape (debug)")
        st.write("Provided feature vector length:", X.shape[1])
        st.write("Feature names and values (truncated):", {k: input_dict[k] for k in list(input_dict.keys())[:10]})
        raise
    return pred[0]

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🍃 Sugarcane Disease Detection (Friendly Qs + YOLO → TabNet)")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    detected_classes, combined_confidence, annotated_img1, annotated_img2, detections_info = run_yolo_and_annotate(img_path)

    st.subheader("🔎 YOLO Detections")
    if detections_info:
        st.table(detections_info)
    else:
        st.write("No detections from YOLO models.")

    st.subheader("📷 Annotated Outputs")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("YOLO Model 1")
        st.image(Image.open(annotated_img1), use_column_width=True)
    with col2:
        st.caption("YOLO Model 2")
        st.image(Image.open(annotated_img2), use_column_width=True)

    if not detected_classes:
        st.warning("No disease detected by YOLO. You can still answer questions, but results may be less reliable.")
        st.stop()

    # Normalize and choose top
    normalized_detected = [normalize_disease_name(d) for d in detected_classes]
    top_class_raw = max(combined_confidence.keys(), key=lambda k: combined_confidence[k])
    top_class_norm = normalize_disease_name(top_class_raw)

    candidate_diseases = [d for d in set(normalized_detected) if d in visual_questions]
    if len(candidate_diseases) == 0:
        st.error("Detected classes do not match configured diseases.")
        st.stop()

    if top_class_norm in candidate_diseases:
        disease_key = top_class_norm
        st.info(f"Auto-selected disease based on model confidence: **{disease_key}**")
    else:
        disease_key = candidate_diseases[0]
        st.info(f"Selected disease: **{disease_key}** (fallback)")

    # Build YOLO visual features for the chosen disease
    yolo_visual_inputs = build_feature_vector_from_yolo(disease_key, detected_classes)

    # Ask friendly questions mapped to technical keys
    st.subheader(f"Questions for: {disease_key}")
    friendly_map = friendly_contextual_questions.get(disease_key, {})
    farmer_answers_map = {}
    if not friendly_map:
        st.write("No contextual questions configured for this disease.")
    else:
        for friendly_text, technical_key in friendly_map.items():
            ans = st.radio(friendly_text, options=["Yes", "No"], key=technical_key)
            farmer_answers_map[technical_key] = (ans == "Yes")

    if st.button("Predict Disease Presence"):
        pred = predict_disease_with_answers(disease_key, farmer_answers_map, yolo_visual_inputs)
        if pred == 1:
            st.success(f"Disease **{disease_key}** is predicted to be **PRESENT**.")
        else:
            st.info(f"Disease **{disease_key}** is predicted to be **ABSENT**.")
