import numpy as np
from typing import List, Optional
import logging
import cv2

logger = logging.getLogger(__name__)

# Lazy loaded model
_face_app = None


class FaceEmbedder:
    def __init__(self):
        self.app = None

    def _load_model(self):
        """Lazy load InsightFace model."""
        if self.app is None:
            try:
                from insightface.app import FaceAnalysis

                # Use CoreML on macOS for better performance
                providers = ['CPUExecutionProvider']
                try:
                    import platform
                    if platform.system() == 'Darwin':
                        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                except:
                    pass

                self.app = FaceAnalysis(
                    name="buffalo_l",
                    providers=providers
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading InsightFace: {e}")
                raise

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-dimensional embedding from a face image.

        Args:
            face_image: BGR image containing a face

        Returns:
            512-dim embedding vector or None if no face detected
        """
        self._load_model()

        # Ensure image is in correct format
        if face_image is None or face_image.size == 0:
            return None

        # Resize if too small
        h, w = face_image.shape[:2]
        if h < 112 or w < 112:
            scale = max(112 / h, 112 / w)
            face_image = cv2.resize(face_image, None, fx=scale, fy=scale)

        try:
            faces = self.app.get(face_image)
            if len(faces) == 0:
                logger.debug("No face detected in image")
                return None

            # Return embedding of the largest face
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return largest_face.embedding
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

    def get_embeddings_from_images(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Get average embedding from multiple reference images.

        Args:
            images: List of BGR images

        Returns:
            Average embedding vector or None if no faces found
        """
        embeddings = []

        for img in images:
            emb = self.get_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.warning("No faces detected in any reference images")
            return None

        # Return normalized average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        return avg_embedding

    def detect_and_embed(self, image: np.ndarray) -> List[dict]:
        """
        Detect all faces in image and return their embeddings.

        Args:
            image: BGR image

        Returns:
            List of dicts with bbox and embedding
        """
        self._load_model()

        try:
            faces = self.app.get(image)
            results = []

            for face in faces:
                bbox = face.bbox.astype(int).tolist()
                embedding = face.embedding

                results.append({
                    "bbox": bbox,
                    "embedding": embedding,
                    "det_score": float(face.det_score)
                })

            return results
        except Exception as e:
            logger.error(f"Error in detect_and_embed: {e}")
            return []
