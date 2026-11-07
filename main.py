"""Interface Streamlit pour creer un panorama avec SIFT."""

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from panorama_processor import (
    build_panorama_pipeline,
    compute_homography,
    create_panorama,
    detect_sift_features,
    match_features,
)


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    """Convertit un fichier envoye par Streamlit en image OpenCV BGR."""
    try:
        pil_image = Image.open(uploaded_file)
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    except Exception as error:
        raise ValueError("Impossible de lire cette image. Utilisez JPG, JPEG ou PNG.") from error

    rgb_image = np.array(pil_image)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convertit une image OpenCV BGR en RGB pour Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_png(image: np.ndarray) -> bytes:
    """Encode une image BGR en PNG pour le bouton de telechargement."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Impossible de preparer le panorama pour le telechargement.")
    return encoded.tobytes()


def add_log(message: str) -> None:
    """Ajoute une ligne dans le journal affiche par l'interface."""
    st.session_state.logs.append(message)


def reset_results() -> None:
    """Efface les anciens resultats lorsque les images changent."""
    st.session_state.features1 = None
    st.session_state.features2 = None
    st.session_state.match_result = None
    st.session_state.homography_result = None
    st.session_state.panorama = None
    st.session_state.result_image = None
    st.session_state.result_caption = ""


def init_state() -> None:
    """Initialise les variables conservees par Streamlit."""
    defaults = {
        "features1": None,
        "features2": None,
        "match_result": None,
        "homography_result": None,
        "panorama": None,
        "result_image": None,
        "result_caption": "",
        "logs": [],
        "last_file_names": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def require_images(image1, image2) -> None:
    """Verifie que les deux images sont disponibles."""
    if image1 is None or image2 is None:
        raise ValueError("Veuillez charger les deux images avant de lancer ce traitement.")


def detect_keypoints(image1: np.ndarray, image2: np.ndarray) -> None:
    """Detecte et affiche les points cles SIFT."""
    require_images(image1, image2)
    features1 = detect_sift_features(image1)
    features2 = detect_sift_features(image2)

    st.session_state.features1 = features1
    st.session_state.features2 = features2
    st.session_state.match_result = None
    st.session_state.homography_result = None
    st.session_state.panorama = None
    st.session_state.result_image = None
    st.session_state.result_caption = ""

    add_log(f"Points cles image 1 : {len(features1.keypoints)}")
    add_log(f"Points cles image 2 : {len(features2.keypoints)}")


def show_matches(image1: np.ndarray, image2: np.ndarray) -> None:
    """Calcule puis affiche les correspondances fiables."""
    require_images(image1, image2)
    if st.session_state.features1 is None or st.session_state.features2 is None:
        detect_keypoints(image1, image2)

    match_result = match_features(
        image1,
        image2,
        st.session_state.features1,
        st.session_state.features2,
    )
    st.session_state.match_result = match_result
    st.session_state.homography_result = None
    st.session_state.panorama = None
    st.session_state.result_image = match_result.visualization
    st.session_state.result_caption = "Correspondances conservees apres le test de Lowe"
    add_log(f"Correspondances apres ratio de Lowe : {match_result.raw_match_count}")
    add_log(f"Correspondances symetriques : {match_result.mutual_match_count}")
    add_log(f"Correspondances fiables utilisees : {len(match_result.good_matches)}")


def generate_panorama(image1: np.ndarray, image2: np.ndarray) -> None:
    """Execute le pipeline complet et affiche le panorama final."""
    require_images(image1, image2)

    if st.session_state.features1 is None or st.session_state.features2 is None:
        (
            features1,
            features2,
            match_result,
            homography_result,
            panorama,
        ) = build_panorama_pipeline(image1, image2)
        st.session_state.features1 = features1
        st.session_state.features2 = features2
        st.session_state.match_result = match_result
        st.session_state.homography_result = homography_result
        st.session_state.panorama = panorama
    else:
        if st.session_state.match_result is None:
            st.session_state.match_result = match_features(
                image1,
                image2,
                st.session_state.features1,
                st.session_state.features2,
            )

        st.session_state.homography_result = compute_homography(
            image1,
            image2,
            st.session_state.features1,
            st.session_state.features2,
            st.session_state.match_result.good_matches,
        )
        st.session_state.panorama = create_panorama(
            image1,
            image2,
            st.session_state.homography_result.matrix,
        )

    st.session_state.result_image = st.session_state.panorama
    st.session_state.result_caption = "Panorama final"
    add_log(f"Correspondances fiables utilisees : {len(st.session_state.match_result.good_matches)}")
    add_log(f"Inliers RANSAC : {st.session_state.homography_result.inlier_count}")
    add_log(f"Ratio inliers : {st.session_state.homography_result.inlier_ratio:.2f}")
    add_log(f"Erreur moyenne : {st.session_state.homography_result.mean_reprojection_error:.2f} px")
    add_log(
        "Panorama genere : "
        f"{st.session_state.panorama.shape[1]} x {st.session_state.panorama.shape[0]} pixels"
    )


def render_uploaded_images(image1, image2) -> None:
    """Affiche les deux images sources."""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image 1")
        if image1 is not None:
            st.image(bgr_to_rgb(image1), use_container_width=True)
        else:
            st.info("Chargez la premiere image.")

    with col2:
        st.subheader("Image 2")
        if image2 is not None:
            st.image(bgr_to_rgb(image2), use_container_width=True)
        else:
            st.info("Chargez la deuxieme image.")


def render_keypoint_images() -> None:
    """Affiche les visualisations des points cles si elles existent."""
    if st.session_state.features1 is None or st.session_state.features2 is None:
        return

    st.subheader("Points cles detectes")
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            bgr_to_rgb(st.session_state.features1.visualization),
            caption=f"Image 1 : {len(st.session_state.features1.keypoints)} points cles",
            use_container_width=True,
        )
    with col2:
        st.image(
            bgr_to_rgb(st.session_state.features2.visualization),
            caption=f"Image 2 : {len(st.session_state.features2.keypoints)} points cles",
            use_container_width=True,
        )


def render_result() -> None:
    """Affiche les correspondances ou le panorama final."""
    st.subheader("Resultat")
    if st.session_state.result_image is None:
        st.info("Le resultat apparaitra ici apres un traitement.")
        return

    st.image(
        bgr_to_rgb(st.session_state.result_image),
        caption=st.session_state.result_caption,
        use_container_width=True,
    )

    if st.session_state.panorama is not None:
        png_data = encode_png(st.session_state.panorama)
        st.download_button(
            "Telecharger le panorama",
            data=png_data,
            file_name="panorama_sift.png",
            mime="image/png",
        )


def render_logs() -> None:
    """Affiche le journal des traitements."""
    st.subheader("Journal")
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs), language="text")
    else:
        st.code("Application prete. Chargez deux images qui se chevauchent.", language="text")

def main() -> None:
    st.set_page_config(
        page_title="Creation de panorama avec SIFT",
        layout="wide",
    )
    init_state()

    st.title("Creation de panorama avec SIFT")
    st.caption("Version amelioree : SIFT contraste, matching symetrique, RANSAC robuste et fusion progressive.")

    with st.sidebar:
        st.header("Images")
        uploaded_image1 = st.file_uploader(
            "Charger image 1",
            type=["jpg", "jpeg", "png"],
            key="image1_uploader",
        )
        uploaded_image2 = st.file_uploader(
            "Charger image 2",
            type=["jpg", "jpeg", "png"],
            key="image2_uploader",
        )

        file_names = (
            uploaded_image1.name if uploaded_image1 else None,
            uploaded_image2.name if uploaded_image2 else None,
        )
        if file_names != st.session_state.last_file_names:
            reset_results()
            st.session_state.logs = []
            st.session_state.last_file_names = file_names

        st.header("Traitement")
        detect_button = st.button("Detecter les points cles", use_container_width=True)
        match_button = st.button("Afficher les correspondances", use_container_width=True)
        panorama_button = st.button("Generer le panorama", use_container_width=True)

    image1 = image2 = None
    try:
        if uploaded_image1 is not None:
            image1 = decode_uploaded_image(uploaded_image1)
        if uploaded_image2 is not None:
            image2 = decode_uploaded_image(uploaded_image2)
    except Exception as error:
        st.error(str(error))
        add_log("Erreur : " + str(error))

    if detect_button or match_button or panorama_button:
        try:
            with st.spinner("Traitement en cours..."):
                if detect_button:
                    detect_keypoints(image1, image2)
                elif match_button:
                    show_matches(image1, image2)
                elif panorama_button:
                    generate_panorama(image1, image2)
        except Exception as error:
            st.error(str(error))
            add_log("Erreur : " + str(error))

    render_uploaded_images(image1, image2)
    render_keypoint_images()
    render_result()
    render_logs()


if __name__ == "__main__":
    main()
