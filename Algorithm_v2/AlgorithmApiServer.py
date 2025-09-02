import argparse
import sys
import numpy as np
import base64
import json
import cv2
import time
from flask import Flask, request, jsonify
from lib.anomaly_detect import AnomalyDetector
from lib.checkface import FaceFeatureExtractor
from lib.OpenVinoYoloV5Detector import OpenVinoYoloV5Detector
from concurrent.futures import ThreadPoolExecutor
import os

# ---------- 关键补丁：递归转 Python 原生类型 ----------
def _to_native(x):
    import numpy as _np
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, _np.ndarray):
        return x.tolist()
    return x
# -----------------------------------------------------

# 异步线程池（仍保留线程池以防阻塞）
executor = ThreadPoolExecutor(max_workers=1)

sys.path = [p for p in sys.path if "insightface-master" not in p]

app = Flask(__name__)

@app.route("/image/objectDetect", methods=['POST'])
def imageObjectDetect():
    data = {
        "code": 0,
        "msg": "unknown error",
    }

    t_total_start = time.time()

    try:
        # silent=True 防止非 JSON 报 400
        params = request.get_json(silent=True) or request.form
    except:
        params = request.form

    algorithm_str = params.get("algorithm")
    image_base64 = params.get("image_base64", None)

    if image_base64:
        if algorithm_str in ["face_recognition"]:
            # 解码图像
            t_decode_start = time.time()
            encoded_image_byte = base64.b64decode(image_base64)
            image_array = np.frombuffer(encoded_image_byte, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            t_decode_end = time.time()

            if image is None:
                data["msg"] = "image decode failed"
                return jsonify(data)

            if algorithm_str == "face_recognition":
                # 执行目标检测
                t_detect_start = time.time()
                detect_num, detect_data = openVinoYoloV5Detector.detect(image)
                t_detect_end = time.time()

                # 提取人物框
                person_boxes = [
                    {
                        "x1": int(item["location"]["x1"]),
                        "y1": int(item["location"]["y1"]),
                        "x2": int(item["location"]["x2"]),
                        "y2": int(item["location"]["y2"])
                    }
                    for item in detect_data if item.get("class_name") == "person"
                ][:4]
                print(f"[DEBUG] 检测到 {len(person_boxes)} 个 person")

                # 异步提取人脸（整帧 + Center-in-Box + 防抖 已在类内实现）
                try:
                    t_face_start = time.time()
                    future = executor.submit(face_feature_extractor.extract_faces, image, person_boxes)
                    face_results = future.result(timeout=3.0)   # 放宽到 3s 更稳
                    t_face_end = time.time()
                    print(f"[DEBUG] 人脸提取耗时: {(t_face_end - t_face_start)*1000:.2f} ms")
                except Exception as e:
                    print(f"[ERROR] 人脸提取异常: {e}")
                    face_results = []

                # 关键：在放入 data 前做一次原生化，避免 numpy 类型
                data["result"] = _to_native({
                    "detect_num": detect_num,
                    "detect_data": detect_data,
                    "face_features": face_results
                })

            data["code"] = 1000
            data["msg"] = "success"

        elif algorithm_str == "anomaly_autoencoder":
            encoded_image_byte = base64.b64decode(image_base64)
            image_array = np.frombuffer(encoded_image_byte, np.uint8)
            image_np = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image_np is None:
                return jsonify({"code": 0, "msg": "image decode failed", "result": {}})

            if 'anomaly_detector' not in globals():
                global anomaly_detector
                anomaly_detector = AnomalyDetector()

            result = anomaly_detector.update_and_detect(image_np)
            print(f"[DEBUG] happen_score = {result.get('score', 0):.4f}, ready = {result.get('ready', False)}")

            return jsonify({
                "code": 1000,
                "msg": "ok",
                "result": {
                    "detect_data": [],  # 不包含框
                    "face_features": [],
                    "happen_score": result["score"] if result.get("ready") else 0.0
                }
            })

        else:
            data["msg"] = f"algorithm={algorithm_str} not supported"
    else:
        data["msg"] = "image not uploaded"

    t_total_end = time.time()
    print(algorithm_str)
    print(f"[DEBUG] 总耗时: {(t_total_end - t_total_start)*1000:.2f} ms")

    # 用 jsonify 固定 JSON 响应头
    return jsonify(data)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", type=int, default=0)
    parse.add_argument("--processes", type=int, default=1)
    parse.add_argument("--port", type=int, default=9003)
    parse.add_argument("--weights", type=str, default="weights")

    flags, _ = parse.parse_known_args(sys.argv[1:])

    # ---------- InsightFace 初始化（保持你的参数） ----------
    def _warmup_face_extractor(fe, w=1280, h=720):
        dummy = np.zeros((h, w, 3), np.uint8)
        fe.extract_faces(dummy, [{"x1": 0, "y1": 0, "x2": w - 1, "y2": h - 1}])

    try:
        face_feature_extractor = FaceFeatureExtractor(
            det_size=(800, 800),     # 小脸更稳，需要更快可改回 (640,640)
            ctx_id=0,                # 0=GPU（有 onnxruntime-gpu 时启用），无GPU会进 except 回落
            person_box_margin=0.12,  # 归属时外扩，减少“头探出框外”
            # 防抖参数（你的类里已默认启用）
            kps_smooth_alpha=0.6,
            bbox_smooth_alpha=0.6,
            enable_kps_smooth=True,
            enable_feat_agg=True,
            feat_bank_len=30,
            quality_gate=True,      # 联调期关质量门控，确保有数据
            quality_thresh=0.35
        )
    except Exception as _:
        face_feature_extractor = FaceFeatureExtractor(
            det_size=(800, 800),
            ctx_id=-1,               # 回落 CPU
            person_box_margin=0.12,
            kps_smooth_alpha=0.6,
            bbox_smooth_alpha=0.6,
            enable_kps_smooth=True,
            enable_feat_agg=True,
            feat_bank_len=30,
            quality_gate=True,
            quality_thresh=0.35
        )

    _warmup_face_extractor(face_feature_extractor)

    # ---------- OpenVINO 检测器（保持你的原配置） ----------
    openVinoYoloV5Detector_IN_conf = {
        "weight_file": "weights/yolov5n_openvino_model/yolov5n.xml",
        "device": "GPU"   # 若无 Intel GPU 可改为 "CPU"
    }
    openVinoYoloV5Detector = OpenVinoYoloV5Detector(IN_conf=openVinoYoloV5Detector_IN_conf)

    app.run(host="0.0.0.0", port=flags.port, debug=(flags.debug == 1))
