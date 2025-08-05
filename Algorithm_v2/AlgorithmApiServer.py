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
        params = request.get_json()
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

            if algorithm_str == "face_recognition":
                # 执行目标检测
                t_detect_start = time.time()
                detect_num, detect_data = openVinoYoloV5Detector.detect(image)
                t_detect_end = time.time()

                # 提取人物框
                person_boxes = [
                    {
                        "x1": item["location"]["x1"],
                        "y1": item["location"]["y1"],
                        "x2": item["location"]["x2"],
                        "y2": item["location"]["y2"]
                    }
                    for item in detect_data if item["class_name"] == "person"
                ][:4]
                print(f"[DEBUG] 检测到 {len(person_boxes)} 个 person")

                # 异步提取人脸
                try:
                    t_face_start = time.time()
                    future = executor.submit(face_feature_extractor.extract_faces, image, person_boxes)
                    face_results = future.result(timeout=2.0)
                    t_face_end = time.time()
                    print(f"[DEBUG] 人脸提取耗时: {(t_face_end - t_face_start)*1000:.2f} ms")

                except Exception as e:
                    print(f"[ERROR] 人脸提取异常: {e}")
                    face_results = []

                data["result"] = {
                    "detect_num": detect_num,
                    "detect_data": detect_data,
                    "face_features": face_results
                }

            data["code"] = 1000
            data["msg"] = "success"

        elif algorithm_str == "anomaly_autoencoder":
            encoded_image_byte = base64.b64decode(image_base64)
            image_array = np.frombuffer(encoded_image_byte, np.uint8)
            image_np = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if 'anomaly_detector' not in globals():
                global anomaly_detector
                anomaly_detector = AnomalyDetector()

            result = anomaly_detector.update_and_detect(image_np)
            print(f"[DEBUG] happen_score = {result['score']:.4f}, ready = {result['ready']}")

            return jsonify({
                "code": 1000,
                "msg": "ok",
                "result": {
                    "detect_data": [],  # 不包含框
                    "face_features": [],
                    "happen_score": result["score"] if result["ready"] else 0.0
                }
            })

        else:
            data["msg"] = f"algorithm={algorithm_str} not supported"
    else:
        data["msg"] = "image not uploaded"

    t_total_end = time.time()
    print(algorithm_str)
    print(f"[DEBUG] 总耗时: {(t_total_end - t_total_start)*1000:.2f} ms")

    # 保存响应结果到本地文件
    with open("debug_result.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return json.dumps(data, ensure_ascii=False)



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", type=int, default=0)
    parse.add_argument("--processes", type=int, default=1)
    parse.add_argument("--port", type=int, default=9003)
    parse.add_argument("--weights", type=str, default="weights")

    flags, _ = parse.parse_known_args(sys.argv[1:])

    face_feature_extractor = FaceFeatureExtractor()

    openVinoYoloV5Detector_IN_conf = {
        "weight_file": "weights/yolov5n_openvino_model/yolov5n.xml",
        "device": "GPU"
    }
    openVinoYoloV5Detector = OpenVinoYoloV5Detector(IN_conf=openVinoYoloV5Detector_IN_conf)

    app.run(host="0.0.0.0", port=flags.port, debug=(flags.debug == 1))
