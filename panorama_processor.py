"""Fonctions de traitement d'image pour creer un panorama avec SIFT."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


MIN_KEYPOINTS = 20
MIN_MATCHES = 12
MIN_INLIERS = 20
MAX_CANVAS_PIXELS = 100_000_000


@dataclass
class FeatureResult:
    """Resultat de la detection SIFT pour une image."""

    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    visualization: np.ndarray


@dataclass
class MatchResult:
    """Resultat de la recherche de correspondances."""

    good_matches: List[cv2.DMatch]
    visualization: np.ndarray
    raw_match_count: int
    mutual_match_count: int


@dataclass
class HomographyResult:
    """Homographie et indicateurs de qualite selon RANSAC."""

    matrix: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    inlier_ratio: float
    mean_reprojection_error: float
    visualization: np.ndarray


def load_image(path: str) -> np.ndarray:
    """Charge une image avec OpenCV et verifie qu'elle est utilisable."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Impossible de charger l'image selectionnee.")
    return image


def _prepare_gray_for_sift(image: np.ndarray) -> np.ndarray:
    """Prepare une image en niveaux de gris plus contrastee pour SIFT."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def detect_sift_features(image: np.ndarray) -> FeatureResult:
    """Detecte les points cles SIFT et calcule leurs descripteurs."""
    gray_image = _prepare_gray_for_sift(image)
    sift = cv2.SIFT_create(
        nfeatures=8000,
        contrastThreshold=0.02,
        edgeThreshold=12,
        sigma=1.6,
    )
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    if len(keypoints) < MIN_KEYPOINTS or descriptors is None:
        raise ValueError(
            "Pas assez de points cles detectes. Essayez une image plus texturee."
        )

    visualization = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return FeatureResult(keypoints, descriptors, visualization)


def _ratio_matches(descriptors1, descriptors2, ratio_threshold: float) -> List[cv2.DMatch]:
    """Applique le test de ratio de Lowe dans un sens."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue
        best_match, second_match = match_pair
        if best_match.distance < ratio_threshold * second_match.distance:
            good_matches.append(best_match)
    return good_matches


def _mutual_matches(matches_12: List[cv2.DMatch], matches_21: List[cv2.DMatch]) -> List[cv2.DMatch]:
    """Garde seulement les correspondances validees dans les deux sens."""
    reverse_pairs = {(match.trainIdx, match.queryIdx) for match in matches_21}
    return [
        match for match in matches_12
        if (match.queryIdx, match.trainIdx) in reverse_pairs
    ]


def match_features(
    image1: np.ndarray,
    image2: np.ndarray,
    features1: FeatureResult,
    features2: FeatureResult,
    ratio_threshold: float = 0.70,
) -> MatchResult:
    """Cherche les correspondances fiables avec Lowe + validation symetrique."""
    matches_12 = _ratio_matches(features1.descriptors, features2.descriptors, ratio_threshold)
    matches_21 = _ratio_matches(features2.descriptors, features1.descriptors, ratio_threshold)
    good_matches = _mutual_matches(matches_12, matches_21)

    # Si la validation symetrique est trop severe, on garde le ratio test simple.
    if len(good_matches) < MIN_MATCHES and len(matches_12) >= MIN_MATCHES:
        good_matches = matches_12

    good_matches = sorted(good_matches, key=lambda match: match.distance)

    if len(good_matches) < MIN_MATCHES:
        raise ValueError(
            f"Pas assez de correspondances fiables ({len(good_matches)} trouvees)."
        )

    visualization = cv2.drawMatches(
        image1,
        features1.keypoints,
        image2,
        features2.keypoints,
        good_matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return MatchResult(
        good_matches=good_matches,
        visualization=visualization,
        raw_match_count=len(matches_12),
        mutual_match_count=len(_mutual_matches(matches_12, matches_21)),
    )


def _points_from_matches(
    features1: FeatureResult,
    features2: FeatureResult,
    matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrait les points image 1 et image 2 des correspondances."""
    points_image1 = np.float32(
        [features1.keypoints[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    points_image2 = np.float32(
        [features2.keypoints[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    return points_image1, points_image2


def _mean_reprojection_error(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    homography: np.ndarray,
    inlier_mask: np.ndarray,
) -> float:
    """Calcule l'erreur moyenne sur les inliers RANSAC."""
    projected = cv2.perspectiveTransform(source_points, homography)
    errors = np.linalg.norm(projected - destination_points, axis=2).ravel()
    inliers = inlier_mask.ravel().astype(bool)
    if not inliers.any():
        return float("inf")
    return float(errors[inliers].mean())


def _find_homography_candidate(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    reprojection_threshold: float,
):
    """Calcule une homographie robuste avec USAC si disponible, sinon RANSAC."""
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        method,
        reprojection_threshold,
    )
    if homography is None or inlier_mask is None:
        return None

    inlier_count = int(inlier_mask.ravel().sum())
    inlier_ratio = inlier_count / max(len(source_points), 1)
    error = _mean_reprojection_error(source_points, destination_points, homography, inlier_mask)
    return homography, inlier_mask, inlier_count, inlier_ratio, error


def _score_homography(inlier_count: int, inlier_ratio: float, error: float) -> float:
    """Score simple : beaucoup d'inliers, ratio eleve, erreur faible."""
    return inlier_count * inlier_ratio / max(error, 0.25)


def compute_homography(
    image1: np.ndarray,
    image2: np.ndarray,
    features1: FeatureResult,
    features2: FeatureResult,
    matches: List[cv2.DMatch],
    reprojection_threshold: float = 4.0,
) -> HomographyResult:
    """Calcule la meilleure homographie qui projette l'image 2 vers l'image 1."""
    points_image1, points_image2 = _points_from_matches(features1, features2, matches)

    # Direction normale : image 2 -> image 1.
    normal = _find_homography_candidate(
        points_image2,
        points_image1,
        reprojection_threshold,
    )
    # Direction inverse estimee separement, puis inversee pour garder image 2 -> image 1.
    inverse = _find_homography_candidate(
        points_image1,
        points_image2,
        reprojection_threshold,
    )

    candidates = []
    if normal is not None:
        homography, mask, count, ratio, error = normal
        candidates.append((homography, mask, count, ratio, error))
    if inverse is not None:
        homography_12, mask, count, ratio, error = inverse
        inverted = np.linalg.inv(homography_12)
        candidates.append((inverted, mask, count, ratio, error))

    if not candidates:
        raise ValueError("L'homographie n'a pas pu etre calculee.")

    homography, inlier_mask, inlier_count, inlier_ratio, error = max(
        candidates,
        key=lambda item: _score_homography(item[2], item[3], item[4]),
    )

    if inlier_count < MIN_INLIERS:
        raise ValueError(
            f"Homographie peu fiable : seulement {inlier_count} inliers."
        )
    if inlier_ratio < 0.12:
        raise ValueError(
            f"Homographie peu fiable : ratio d'inliers trop faible ({inlier_ratio:.2f})."
        )
    if error > 5.0:
        raise ValueError(
            f"Alignement peu fiable : erreur moyenne de reprojection {error:.2f} px."
        )

    visualization = cv2.drawMatches(
        image1,
        features1.keypoints,
        image2,
        features2.keypoints,
        matches,
        None,
        matchesMask=inlier_mask.ravel().tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return HomographyResult(
        matrix=homography,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        mean_reprojection_error=error,
        visualization=visualization,
    )


def _image_corners(image: np.ndarray) -> np.ndarray:
    """Retourne les quatre coins d'une image au format attendu par perspectiveTransform."""
    height, width = image.shape[:2]
    return np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)


def _distance_weight(mask: np.ndarray) -> np.ndarray:
    """Cree un poids progressif eleve au centre de la zone valide."""
    mask_uint8 = mask.astype(np.uint8)
    distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if distance.max() > 0:
        distance = distance / distance.max()
    return distance.astype(np.float32)


def _crop_to_content(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Recadre le panorama autour des pixels reellement remplis."""
    y_values, x_values = np.where(mask)
    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError("Le panorama obtenu est vide.")
    cropped = image[
        y_values.min() : y_values.max() + 1,
        x_values.min() : x_values.max() + 1,
    ]
    if cropped.size == 0 or np.count_nonzero(cropped) == 0:
        raise ValueError("Le panorama obtenu semble entierement noir.")
    return cropped


def create_panorama(
    image1: np.ndarray,
    image2: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    """Projette les deux images dans un canevas commun et fusionne progressivement."""
    corners_image1 = _image_corners(image1)
    corners_image2 = _image_corners(image2)
    transformed_corners_image2 = cv2.perspectiveTransform(corners_image2, homography)

    all_corners = np.concatenate((corners_image1, transformed_corners_image2), axis=0)
    min_x, min_y = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    output_width = max_x - min_x
    output_height = max_y - min_y
    if output_width <= 0 or output_height <= 0:
        raise ValueError("Dimensions invalides pour le panorama.")
    if output_width * output_height > MAX_CANVAS_PIXELS:
        raise ValueError("Le panorama calcule est trop grand ou incoherent.")

    translation = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]],
        dtype=np.float64,
    )

    warped_image2 = cv2.warpPerspective(
        image2,
        translation @ homography,
        (output_width, output_height),
    )
    mask2_source = np.full(image2.shape[:2], 255, dtype=np.uint8)
    mask2 = cv2.warpPerspective(
        mask2_source,
        translation @ homography,
        (output_width, output_height),
    ) > 0

    canvas_image1 = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    mask1 = np.zeros((output_height, output_width), dtype=bool)
    x_offset = -min_x
    y_offset = -min_y
    height1, width1 = image1.shape[:2]
    canvas_image1[y_offset : y_offset + height1, x_offset : x_offset + width1] = image1
    mask1[y_offset : y_offset + height1, x_offset : x_offset + width1] = True

    union_mask = mask1 | mask2
    if not union_mask.any():
        raise ValueError("Le panorama obtenu est vide.")

    weight1 = _distance_weight(mask1)
    weight2 = _distance_weight(mask2)
    total_weight = weight1 + weight2
    total_weight[total_weight == 0] = 1.0

    panorama_float = (
        canvas_image1.astype(np.float32) * weight1[..., None]
        + warped_image2.astype(np.float32) * weight2[..., None]
    ) / total_weight[..., None]
    panorama = np.clip(panorama_float, 0, 255).astype(np.uint8)
    return _crop_to_content(panorama, union_mask)


def build_panorama_pipeline(
    image1: np.ndarray,
    image2: np.ndarray,
) -> Tuple[FeatureResult, FeatureResult, MatchResult, HomographyResult, np.ndarray]:
    """Execute toutes les etapes principales du pipeline."""
    features1 = detect_sift_features(image1)
    features2 = detect_sift_features(image2)
    match_result = match_features(image1, image2, features1, features2)
    homography_result = compute_homography(
        image1,
        image2,
        features1,
        features2,
        match_result.good_matches,
    )
    panorama = create_panorama(image1, image2, homography_result.matrix)
    return features1, features2, match_result, homography_result, panorama
