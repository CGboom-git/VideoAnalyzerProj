from insightface.app import FaceAnalysis
import os
import numpy as np
import cv2
from collections import defaultdict

def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)

class _Ema:
    def __init__(self, alpha=0.6):
        self.alpha = float(alpha)
        self.prev = None
    def __call__(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        if self.prev is None:
            self.prev = arr
        else:
            self.prev = self.alpha * self.prev + (1 - self.alpha) * arr
        return self.prev

class _FeatBank:
    def __init__(self, maxlen=20):
        self.feats = []
        self.weights = []
        self.maxlen = int(maxlen)
    def push(self, feat, w=1.0):
        f = np.asarray(feat, dtype=np.float32)
        self.feats.append(f); self.weights.append(float(max(1e-3, w)))
        if len(self.feats) > self.maxlen:
            self.feats.pop(0); self.weights.pop(0)
    def centroid(self):
        if not self.feats:
            return None
        W = np.asarray(self.weights, dtype=np.float32)
        F = np.vstack(self.feats)  # [T, D]
        c = (F * W[:, None]).sum(0) / (W.sum() + 1e-9)
        return _normalize(c)

def _lap_var(gray):
    return cv2.Laplacian(gray, cv2.CV_32F).var()

def _quality_score(img, bbox):
    x1,y1,x2,y2 = map(int, bbox)
    x1=max(0,x1); y1=max(0,y1); x2=min(img.shape[1]-1,x2); y2=min(img.shape[0]-1,y2)
    if x2<=x1 or y2<=y1: return 0.0
    patch = img[y1:y2, x1:x2]
    if patch.size==0: return 0.0
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blur = _lap_var(g)
    area = (x2-x1) * (y2-y1)
    mean = g.mean()
    s_blur = np.tanh(blur / 200.0)
    s_area = np.tanh(area / 12000.0)
    s_luma = 1.0 - abs(mean - 128.0)/128.0
    return 0.5*s_blur + 0.3*s_area + 0.2*s_luma

class FaceFeatureExtractor:
    def __init__(self,
                 det_size=(640, 640),
                 ctx_id=0,
                 person_box_margin=0.10,
                 kps_smooth_alpha=0.6,
                 bbox_smooth_alpha=0.6,
                 enable_kps_smooth=True,
                 enable_feat_agg=True,
                 feat_bank_len=20,
                 quality_gate=True,
                 quality_thresh=0.35,
                 # 轻量级跟踪配置
                 track_iou_thresh=0.3,
                 track_ttl=10):
        model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
        self.app = FaceAnalysis(name="buffalo_s", root=model_root, allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        self.person_box_margin = float(person_box_margin)

        # 防抖/特征聚合按“稳定 track_id”维护
        self.kps_smoothers = defaultdict(lambda: _Ema(kps_smooth_alpha))
        self.bbox_smoothers = defaultdict(lambda: _Ema(bbox_smooth_alpha))
        self.feat_banks     = defaultdict(lambda: _FeatBank(feat_bank_len))

        self.enable_kps_smooth = bool(enable_kps_smooth)
        self.enable_feat_agg   = bool(enable_feat_agg)
        self.quality_gate      = bool(quality_gate)
        self.quality_thresh    = float(quality_thresh)

        # 简易多人跟踪（给 person_box 稳定 ID）
        self.track_iou_thresh = float(track_iou_thresh)
        self.track_ttl = int(track_ttl)
        self._tracks = {}   # id -> {"box":[x1,y1,x2,y2], "ttl":int}
        self._next_id = 1

        print("[DEBUG] InsightFace initialized.")
        print(f"[DEBUG] det_size={det_size}, ctx_id={ctx_id}, margin={self.person_box_margin}")
        print(f"[DEBUG] kps_smooth={enable_kps_smooth}, feat_agg={enable_feat_agg}, q_gate={quality_gate}")

    @staticmethod
    def _clip(v, lo, hi):
        return max(lo, min(hi, v))

    def _expand_person_box(self, box, width, height):
        x1,y1,x2,y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        w = max(1, x2-x1); h = max(1, y2-y1)
        dx = int(round(w * self.person_box_margin))
        dy = int(round(h * self.person_box_margin))
        ex1 = self._clip(x1 - dx, 0, width - 1)
        ey1 = self._clip(y1 - dy, 0, height - 1)
        ex2 = self._clip(x2 + dx, 0, width - 1)
        ey2 = self._clip(y2 + dy, 0, height - 1)
        return {"x1": ex1, "y1": ey1, "x2": ex2, "y2": ey2}

    def _clip_person_box(self, box, width, height):
        x1 = int(self._clip(box["x1"], 0, width - 1))
        y1 = int(self._clip(box["y1"], 0, height - 1))
        x2 = int(self._clip(box["x2"], 0, width - 1))
        y2 = int(self._clip(box["y2"], 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @staticmethod
    def _bbox_center(b):
        x1,y1,x2,y2 = b
        return (0.5*(x1+x2), 0.5*(y1+y2))

    # ---------------- 简易 IoU 跟踪：给 person_box 稳定 ID ----------------
    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2 = a["x1"],a["y1"],a["x2"],a["y2"]
        bx1,by1,bx2,by2 = b["x1"],b["y1"],b["x2"],b["y2"]
        iw = max(0, min(ax2,bx2) - max(ax1,bx1))
        ih = max(0, min(ay2,by2) - max(ay1,by1))
        inter = iw * ih
        if inter <= 0: return 0.0
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / (ua + 1e-9)

    def _assign_track_ids(self, person_boxes, w, h):
        # 先 clip 一遍
        curr = []
        for b in person_boxes:
            cb = self._clip_person_box(b, w, h)
            if cb is not None:
                curr.append(cb)

        # 现有轨迹减寿命
        for tid in list(self._tracks.keys()):
            self._tracks[tid]["ttl"] -= 1
            if self._tracks[tid]["ttl"] <= 0:
                del self._tracks[tid]

        # 匹配：贪心 IoU（足够了）
        assigned = []
        used_tracks = set()
        for b in curr:
            best_iou, best_tid = 0.0, None
            for tid, t in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou = self._iou({"x1":t["box"][0],"y1":t["box"][1],"x2":t["box"][2],"y2":t["box"][3]}, b)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None and best_iou >= self.track_iou_thresh:
                # 续命&更新框
                self._tracks[best_tid] = {"box":[b["x1"],b["y1"],b["x2"],b["y2"]], "ttl": self.track_ttl}
                used_tracks.add(best_tid)
                assigned.append({"id": best_tid, "box": b})
            else:
                # 新建轨迹
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {"box":[b["x1"],b["y1"],b["x2"],b["y2"]], "ttl": self.track_ttl}
                used_tracks.add(tid)
                assigned.append({"id": tid, "box": b})

        return assigned
    # ----------------------------------------------------------------------

    def extract_faces(self, frame, person_boxes):
        h, w = frame.shape[:2]

        # 1) 整帧做人脸检测
        faces = self.app.get(frame)
        print(f"[DEBUG] Full-frame detected {len(faces)} face(s)")

        # 2) 整帧 -> 中间结构（像素坐标）
        face_items = []
        for f in faces:
            if f.bbox is None or f.kps is None or f.embedding is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in f.bbox.tolist()]
            x1 = self._clip(x1, 0, w - 1); y1 = self._clip(y1, 0, h - 1)
            x2 = self._clip(x2, 0, w - 1); y2 = self._clip(y2, 0, h - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            kps = [(int(round(kp[0])), int(round(kp[1]))) for kp in f.kps]
            emb = _normalize(f.embedding)
            face_items.append({
                "bbox": [x1, y1, x2, y2],
                "kps":  kps,
                "embedding": emb,
                "center": self._bbox_center([x1, y1, x2, y2]),
            })

        results = []

        # 3) 给 person_box 分配稳定 track_id（替代 enumerate 的 pid）
        assigned_persons = self._assign_track_ids(person_boxes, w, h)

        # 4) 逐 person（稳定 id）归属 + 只输出一张最佳脸
        for info in assigned_persons:
            tid = info["id"]            # 稳定 id
            raw_box = info["box"]       # 已裁边
            assign_box = self._expand_person_box(raw_box, w, h)
            ax1, ay1, ax2, ay2 = assign_box["x1"], assign_box["y1"], assign_box["x2"], assign_box["y2"]

            # 收集候选
            candidates = []
            for it in face_items:
                cx, cy = it["center"]
                if not (ax1 <= cx <= ax2 and ay1 <= cy <= ay2):
                    continue
                q = _quality_score(frame, it["bbox"]) if self.quality_gate else 1.0
                if self.quality_gate and q < self.quality_thresh:
                    continue
                bx1, by1, bx2, by2 = it["bbox"]
                area = max(1, (bx2 - bx1) * (by2 - by1))
                candidates.append((it, q, area))

            faces_out = []
            if candidates:
                # 只选一张最佳脸
                candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                it, q_best, _ = candidates[0]

                # 预平滑（用该 track 的 prev）
                use_smooth = self.enable_kps_smooth
                prev_kps = self.kps_smoothers[tid].prev if use_smooth else None
                prev_bbox = self.bbox_smoothers[tid].prev if use_smooth else None
                alpha_k = self.kps_smoothers[tid].alpha if use_smooth else 0.0
                alpha_b = self.bbox_smoothers[tid].alpha if use_smooth else 0.0

                def _preview(prev_arr, curr_arr, alpha):
                    if prev_arr is None:
                        return np.asarray(curr_arr, dtype=np.float32)
                    return alpha * np.asarray(prev_arr, dtype=np.float32) + (1 - alpha) * np.asarray(curr_arr,
                                                                                                     dtype=np.float32)

                if use_smooth:
                    _ = _preview(prev_kps, it["kps"], alpha_k)  # 只为内部稳定，不用于输出
                    _ = _preview(prev_bbox, it["bbox"], alpha_b)

                # === 输出“当前帧原始检测”而非平滑后的 ===
                curr_kps_pairs = it["kps"]  # [(x,y), ...] 当前帧
                curr_bbox_int = [int(round(v)) for v in it["bbox"]]  # [x1,y1,x2,y2] 当前帧

                # 特征聚合（内部使用稳定 id）
                if self.enable_feat_agg:
                    self.feat_banks[tid].push(it["embedding"], w=q_best)
                    cent = self.feat_banks[tid].centroid()
                    emb_out = it["embedding"] if cent is None else cent
                else:
                    emb_out = it["embedding"]

                # 扁平化 kps -> 一维 int 数组（给前端画当前帧位置）
                flat_kps = []
                for x, y in curr_kps_pairs:
                    flat_kps.append(int(round(x)))
                    flat_kps.append(int(round(y)))

                faces_out.append({
                    "id": int(tid),
                    "bbox": curr_bbox_int,  # ← 前端只拿当前帧框
                    "kps": flat_kps,  # ← 前端只拿当前帧关键点
                    "embedding": np.asarray(emb_out, dtype=np.float32).tolist()
                })

                # commit 一次（更新该 track 的 prev，用于下一帧内部稳定）
                if use_smooth:
                    self.kps_smoothers[tid](np.asarray(it["kps"], dtype=np.float32))
                    self.bbox_smoothers[tid](np.asarray(it["bbox"], dtype=np.float32))

            print(f"[DEBUG] PersonBox T{tid} {raw_box} <= {len(faces_out)} face(s)")
            results.append({
                "person_box": {
                    "x1": int(raw_box["x1"]), "y1": int(raw_box["y1"]),
                    "x2": int(raw_box["x2"]), "y2": int(raw_box["y2"]),
                },
                "faces": faces_out
            })

        return results

    def reset_history(self):
        self.kps_smoothers.clear()
        self.bbox_smoothers.clear()
        self.feat_banks.clear()
        self._tracks.clear(); self._next_id = 1
