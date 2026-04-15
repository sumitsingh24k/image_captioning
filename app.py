from pathlib import Path

import streamlit as st
from PIL import Image
from caption_service import CaptionService


DEFAULT_PROJECT_DIR = Path(".")


def main() -> None:
    st.set_page_config(page_title="Image Captioning App", page_icon="🖼️", layout="centered")
    st.title("🖼️ Image Captioning Mini Project")
    st.caption("Custom CNN+LSTM model with optional BLIP fallback.")

    with st.sidebar:
        st.header("Settings")
        project_dir = Path(st.text_input("Project directory", str(DEFAULT_PROJECT_DIR)))
        mode = st.selectbox(
            "Caption backend",
            ["Auto (Custom -> BLIP)", "Custom CNN+LSTM", "BLIP fallback"],
            index=0,
        )

    service = CaptionService(project_dir)
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file is None:
        st.info("Upload an image to generate a caption.")
        return

    uploaded_img = Image.open(uploaded_file)
    st.image(uploaded_img, caption="Uploaded image", use_container_width=True)

    if st.button("Generate Caption", type="primary"):
        with st.spinner("Generating caption..."):
            try:
                if mode == "Custom CNN+LSTM":
                    caption = service.caption(uploaded_img, mode="custom")
                elif mode == "BLIP fallback":
                    caption = service.caption(uploaded_img, mode="blip")
                else:
                    caption = service.caption(uploaded_img, mode="auto")

                st.success(caption)
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)


if __name__ == "__main__":
    main()
