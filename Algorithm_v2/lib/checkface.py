from insightface.app import FaceAnalysis
import cv2
import os
import traceback
import numpy as np

class FaceFeatureExtractor:
    def __init__(self, det_size=(640, 640), ctx_id=0):
        model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
        self.app = FaceAnalysis(name="buffalo_s", root=model_root)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print("[DEBUG] InsightFace initialized with model_root =", model_root)

    def extract_faces(self, frame, person_boxes):
        height, width = frame.shape[:2]
        results = []

        for idx, box in enumerate(person_boxes):
            x1 = max(0, min(width - 1, box["x1"]))
            y1 = max(0, min(height - 1, box["y1"]))
            x2 = max(0, min(width - 1, box["x2"]))
            y2 = max(0, min(height - 1, box["y2"]))

            if x2 <= x1 or y2 <= y1:
                print(f"[WARN] Invalid person box: {box}")
                continue

            cropped = frame[y1:y2, x1:x2].copy()
            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                print(f"[WARN] Cropped image is empty at box {box}")
                continue
            
            face_dicts = []
            try:
                faces = self.app.get(cropped)
                print(f"[DEBUG] Box {idx} => Detected {len(faces)} face(s)")

                for i, f in enumerate(faces):
                    if f.bbox is None or f.kps is None:
                        print(f"[WARN] Face {i} in box {idx} is missing bbox or kps.")
                        continue

                    f_dict = {
                        "bbox": [
                            f.bbox[0] + x1,
                            f.bbox[1] + y1,
                            f.bbox[2] + x1,
                            f.bbox[3] + y1
                        ],
                        "kps": [[kp[0] + x1, kp[1] + y1] for kp in f.kps],
                        "embedding": f.embedding.tolist()
                    }

                    face_dicts.append(f_dict)
                    print(f"[Face {i}] Embedding[:5]: {f.embedding[:5]}")
            except Exception as e:
                traceback.print_exc()
                print(f"[ERROR] Failed to extract faces from box {idx}: {e}")
            if face_dicts:
                results.append({
                    "person_id": idx,           # 标明人物框编号
                    "person_box": box,
                    "faces": face_dicts
                })

        return results
