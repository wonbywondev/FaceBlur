import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FaceMatcher:
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize face matcher.

        Args:
            similarity_threshold: Threshold for considering faces as same person (0-1)
        """
        self.similarity_threshold = similarity_threshold

    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Similarity score (0-100%)
        """
        if emb1 is None or emb2 is None:
            return 0.0

        # Normalize embeddings
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

        # Cosine similarity
        similarity = 1 - cosine(emb1, emb2)

        # Convert to percentage
        return max(0.0, min(100.0, similarity * 100))

    def is_same_person(self, emb1: np.ndarray, emb2: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if two embeddings belong to the same person.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Tuple of (is_same, similarity_score)
        """
        similarity = self.calculate_similarity(emb1, emb2)
        is_same = similarity >= (self.similarity_threshold * 100)
        return is_same, similarity

    def cluster_faces(
        self,
        face_detections: List[Dict],
        reference_embedding: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Cluster detected faces by identity.

        Args:
            face_detections: List of dicts with 'embedding', 'timestamp', 'bbox', 'frame'
            reference_embedding: Optional reference embedding for "self" identification

        Returns:
            List of face clusters with appearances
        """
        clusters = []

        for detection in face_detections:
            if detection.get("embedding") is None:
                continue

            emb = detection["embedding"]
            matched_cluster = None

            # Try to match with existing clusters
            for cluster in clusters:
                is_same, similarity = self.is_same_person(emb, cluster["representative"])
                if is_same:
                    matched_cluster = cluster
                    # Update representative with running average
                    n = len(cluster["appearances"])
                    cluster["representative"] = (
                        cluster["representative"] * n + emb
                    ) / (n + 1)
                    break

            if matched_cluster:
                matched_cluster["appearances"].append({
                    "timestamp": detection["timestamp"],
                    "bbox": detection["bbox"],
                    "frame_number": detection.get("frame_number", 0)
                })
                # Update thumbnail if this detection has better confidence
                if detection.get("det_score", 0) > matched_cluster.get("best_score", 0):
                    matched_cluster["thumbnail_frame"] = detection.get("frame")
                    matched_cluster["best_score"] = detection.get("det_score", 0)
            else:
                # Create new cluster
                cluster_id = f"face-{len(clusters):03d}"
                new_cluster = {
                    "face_id": cluster_id,
                    "representative": emb.copy(),
                    "thumbnail_frame": detection.get("frame"),
                    "best_score": detection.get("det_score", 0),
                    "appearances": [{
                        "timestamp": detection["timestamp"],
                        "bbox": detection["bbox"],
                        "frame_number": detection.get("frame_number", 0)
                    }]
                }
                clusters.append(new_cluster)

        # Calculate similarity to reference for each cluster
        if reference_embedding is not None:
            for cluster in clusters:
                _, similarity = self.is_same_person(
                    cluster["representative"],
                    reference_embedding
                )
                cluster["similarity_to_reference"] = similarity
                cluster["is_reference"] = similarity >= (self.similarity_threshold * 100)
        else:
            for cluster in clusters:
                cluster["similarity_to_reference"] = 0.0
                cluster["is_reference"] = False

        # Sort clusters by first appearance
        clusters.sort(key=lambda c: c["appearances"][0]["timestamp"])

        return clusters

    def merge_appearances(self, appearances: List[Dict], gap_threshold: float = 1.0) -> List[Dict]:
        """
        Merge consecutive appearances into ranges.

        Args:
            appearances: List of individual appearances
            gap_threshold: Maximum gap in seconds to merge

        Returns:
            List of merged appearance ranges
        """
        if not appearances:
            return []

        # Sort by timestamp
        sorted_appearances = sorted(appearances, key=lambda a: a["timestamp"])

        merged = []
        current = {
            "start": sorted_appearances[0]["timestamp"],
            "end": sorted_appearances[0]["timestamp"],
            "bbox": sorted_appearances[0]["bbox"]
        }

        for app in sorted_appearances[1:]:
            if app["timestamp"] - current["end"] <= gap_threshold:
                # Extend current range
                current["end"] = app["timestamp"]
                current["bbox"] = app["bbox"]  # Use latest bbox
            else:
                # Start new range
                merged.append(current)
                current = {
                    "start": app["timestamp"],
                    "end": app["timestamp"],
                    "bbox": app["bbox"]
                }

        merged.append(current)
        return merged
