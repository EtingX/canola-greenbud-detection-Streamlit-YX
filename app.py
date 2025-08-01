import streamlit as st
import os
import zipfile
import tempfile
import shutil
import glob
import random
from PIL import Image
from sahi_predict_yx import predict

# Model path mapping
MODEL_MAP = {
    '640 original': '640_n/weights/best.pt',
    '640 advanced': '640_n_ASFFHead_C3K2_PPA_SPDConv/weights/best.pt',
    '960 original': '960_n/weights/best.pt',
    '960 advanced': '960_ASFFHead_C3K2_PPA_SPDConv/weights/best.pt',
}

# Persistent output folder
PERSISTENT_DIR = "persistent_tmp"
os.makedirs(PERSISTENT_DIR, exist_ok=True)

st.title("Canola Green Bud Detection (YOLOv11 (advanced) + SAHI)")
st.markdown("Upload images or ZIP files, choose a model, and click Run to start inference.")

# File uploader (multi-file enabled)
uploaded_files = st.file_uploader(
    "Upload images or ZIP files", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True
)

# Model selector
selected_model = st.selectbox(
    "Select a model. Advanced ones offer better accuracy but take longer to run.", list(MODEL_MAP.keys())
)

if st.button("Run") and uploaded_files:
    # Clear old runs and cached results
    if os.path.exists("runs"):
        shutil.rmtree("runs")
    if os.path.exists(PERSISTENT_DIR):
        shutil.rmtree(PERSISTENT_DIR)
    os.makedirs(PERSISTENT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir, exist_ok=True)

        # Save uploaded content
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name.endswith(".zip"):
                zip_path = os.path.join(tmpdir, file_name)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.read())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(input_dir)
            else:
                img_path = os.path.join(input_dir, file_name)
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.read())

        # Image file filtering (case-insensitive)
        valid_exts = (".jpg", ".jpeg", ".png")
        image_files = [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.lower().endswith(valid_exts)
        ]
        num_input_images = len(image_files)

        # Estimate processing time
        time_per_image = 30 if "advanced" in selected_model else 15
        estimated_seconds = int(num_input_images * time_per_image)
        estimated_minutes = round(estimated_seconds / 60, 1)

        st.info("\U0001F50D Inference started. Please wait...")
        st.warning(f"\U0001F552 CPU inference. Estimated time: {estimated_minutes} minutes "
                   f"({num_input_images} images, approx. {time_per_image} seconds per image)")

        # Slice size
        slice_size = 960 if "960" in selected_model else 640

        # Run prediction
        predict(
            model_type="ultralytics",
            model_path=MODEL_MAP[selected_model],
            model_device="cpu",
            model_confidence_threshold=0.6,
            source=input_dir,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0.45,
            overlap_width_ratio=0.45,
            export_txt=True
        )

        # Result directories
        result_dir = "runs/predict/exp"
        image_dir = os.path.join(result_dir, "visuals")
        label_dir = os.path.join(result_dir, "labels")

        # Get results
        result_images = sorted(glob.glob(os.path.join(image_dir, "*.[pP][nN][gG]")))
        num_images = len(result_images)
        num_boxes = sum(len(open(txt).readlines()) for txt in glob.glob(os.path.join(label_dir, "*.txt")))

        # Random preview (max 5)
        preview_images = random.sample(result_images, min(5, num_images))

        # ZIP packaging
        zip_output = os.path.join(PERSISTENT_DIR, "canola_detection_results.zip")
        with zipfile.ZipFile(zip_output, "w") as zipf:
            for folder in [label_dir, image_dir]:
                for file_path in glob.glob(os.path.join(folder, "*")):
                    arcname = os.path.join(os.path.basename(folder), os.path.basename(file_path))
                    zipf.write(file_path, arcname=arcname)

        # Save to session state
        st.session_state["zip_path"] = zip_output
        st.session_state["num_images"] = num_images
        st.session_state["num_boxes"] = num_boxes
        st.session_state["preview_images"] = preview_images

# Show results (preserved after refresh)
if "zip_path" in st.session_state:
    st.success("\u2705 Inference complete. Preview and download results below.")
    st.markdown(f"**\U0001F4CA {st.session_state['num_images']} images analyzed, "
                f"{st.session_state['num_boxes']} objects detected.**")

    for img_path in st.session_state["preview_images"]:
        st.image(Image.open(img_path), use_container_width=True)

    with open(st.session_state["zip_path"], "rb") as f:
        st.download_button(
            label="\U0001F4E6 Download all results (labels + visuals)",
            data=f,
            file_name="canola_detection_results.zip",
            mime="application/zip"
        )
