#include "Analyzer.h"
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include "Utils/Request.h"
#include "Utils/Log.h"
#include "Utils/Common.h"
#include "Utils/Base64.h"
#include "Scheduler.h"
#include "Config.h"
#include "Control.h"

#ifndef WIN32
#include <opencv2/opencv.hpp>
#else
#ifndef _DEBUG
#include <turbojpeg.h>
#else
#include <opencv2/opencv.hpp>
#endif
#endif

namespace AVSAnalyzer {

    static bool analy_turboJpeg_compress(int height, int width, int channels, unsigned char* bgr, unsigned char*& out_data, unsigned long* out_size) {
#ifdef WIN32
#ifndef _DEBUG

        tjhandle handle = tjInitCompress();
        if (nullptr == handle) {
            return false;
        }

        //pixel_format : TJPF::TJPF_BGR or other
        const int JPEG_QUALITY = 75;
        int pixel_format = TJPF::TJPF_BGR;
        int pitch = tjPixelSize[pixel_format] * width;
        int ret = tjCompress2(handle, bgr, width, pitch, height, pixel_format,
            &out_data, out_size, TJSAMP_444, JPEG_QUALITY, TJFLAG_FASTDCT);

        tjDestroy(handle);

        if (ret != 0) {
            return false;
        }
        return true;
#endif // !_DEBUG
#endif

        return false;
    }

    static bool analy_compressBgrAndEncodeBase64(int height, int width, int channels, unsigned char* bgr, std::string& out_base64) {
    

#ifdef WIN32
#ifndef _DEBUG
        unsigned char* jpeg_data = nullptr;
        unsigned long  jpeg_size = 0;

        analy_turboJpeg_compress(height, width, channels, bgr, jpeg_data, &jpeg_size);

        if (jpeg_size > 0 && jpeg_data != nullptr) {

            Base64Encode(jpeg_data, jpeg_size, out_base64);

            free(jpeg_data);
            jpeg_data = nullptr;

            return true;
        }
        else {
            return false;
        }

#endif // !_DEBUG

#else
        cv::Mat bgr_image(height, width, CV_8UC3, bgr);

        std::vector<int> quality = { 100 };
        std::vector<uchar> jpeg_data;
        cv::imencode(".jpg", bgr_image, jpeg_data, quality);

        Base64Encode(jpeg_data.data(), jpeg_data.size(), out_base64);

        return true;
#endif //WIN32
    }

    std::string mapBehaviorToAlgorithm(const std::string& behaviorCode) {
        if (behaviorCode == "1") return "face_recognition";
        if (behaviorCode == "2") return "anomaly_autoencoder";
        return "face_recognition";
    }
    //AlgorithmWithApi::AlgorithmWithApi(Config* config):mConfig(config)
    //{
    //    LOGI("");
    //}
    AlgorithmWithApi::AlgorithmWithApi(const std::vector<std::string>& hosts, const std::string& algorithmType)
        : mApiHosts(hosts), mAlgorithmType(algorithmType) {
    }
    AlgorithmWithApi::~AlgorithmWithApi()
    {
        LOGI("");
    }
    bool AlgorithmWithApi::test() {
        std::string response;
        int randIndex = rand() % mConfig->algorithmApiHosts.size();
        std::string host = mConfig->algorithmApiHosts[randIndex];
        std::string url = host + "/image/objectDetect";

        Request request;
        bool ret = request.get(url.data(), response);

        //LOGI("ret=%d,response=%s",ret,response.data());
        return ret;

    }

    //bool AlgorithmWithApi::objectDetect(int height, int width, unsigned char* bgr,
    //    std::vector<AlgorithmDetectObject>& detects,
    //    std::vector<PersonWithFace>& face_features) {
    //    cv::Mat image(height, width, CV_8UC3, bgr);

    //    int64_t t1 = getCurTime();
    //    std::string imageBase64;
    //    analy_compressBgrAndEncodeBase64(image.rows, image.cols, 3, image.data, imageBase64);
    //    int64_t t2 = getCurTime();

    //    int randIndex = rand() % mConfig->algorithmApiHosts.size();
    //    std::string host = mConfig->algorithmApiHosts[randIndex];
    //    std::string url = host + "/image/objectDetect";

    //    Json::Value param;
    //    param["appKey"] = "s84dsd#7hf34r3jsk@fs$d#$dd";
    //    param["algorithm"] = "openvino_yolov5";
    //    param["image_base64"] = imageBase64;
    //    std::string data = param.toStyledString();
    //    param = NULL;

    //    int64_t t3 = getCurTime();
    //    Request request;
    //    std::string response;
    //    bool result = request.post(url.data(), data.data(), response);
    //    int64_t t4 = getCurTime();

    //    if (result) {
    //        result = this->parseObjectDetect(response, detects, face_features);
    //    }

    //    return result;
    //}

    //bool AlgorithmWithApi::parseObjectDetect(std::string& response,
    //    std::vector<AlgorithmDetectObject>& detects,
    //    std::vector<PersonWithFace>& face_features) {
    //    Json::CharReaderBuilder builder;
    //    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());

    //    Json::Value root;
    //    JSONCPP_STRING errs;
    //    bool result = false;

    //    if (reader->parse(response.data(), response.data() + response.size(), &root, &errs) && errs.empty()) {
    //        if (root["code"].isInt() && root["code"].asInt() == 1000) {
    //            Json::Value root_result = root["result"];

    //            Json::Value detect_data = root_result["detect_data"];
    //            for (auto& i : detect_data) {
    //                AlgorithmDetectObject object;
    //                Json::Value loc = i["location"];
    //                object.x1 = loc["x1"].asInt();
    //                object.y1 = loc["y1"].asInt();
    //                object.x2 = loc["x2"].asInt();
    //                object.y2 = loc["y2"].asInt();
    //                object.score = i["score"].asFloat();
    //                object.class_name = i["class_name"].asString();
    //                detects.push_back(object);
    //            }

    //            Json::Value face_json = root_result["face_features"];
    //            for (auto& item : face_json) {
    //                PersonWithFace person_face;

    //                if (item.isMember("person_id"))
    //                    person_face.person_id = item["person_id"].asInt();
    //                Json::Value box = item["person_box"];
    //                person_face.person.x1 = box["x1"].asInt();
    //                person_face.person.y1 = box["y1"].asInt();
    //                person_face.person.x2 = box["x2"].asInt();
    //                person_face.person.y2 = box["y2"].asInt();

    //                Json::Value face_list = item["faces"];
    //                for (auto& f : face_list) {
    //                    FaceFeature ff;
    //                    // ✅ 安全处理 bbox 顺序
    //                    Json::Value bbox_arr = f["bbox"];
    //                    if (bbox_arr.isArray() && bbox_arr.size() == 4) {
    //                        ff.bbox.push_back(bbox_arr[0].asInt());
    //                        ff.bbox.push_back(bbox_arr[1].asInt());
    //                        ff.bbox.push_back(bbox_arr[2].asInt());
    //                        ff.bbox.push_back(bbox_arr[3].asInt());
    //                    }

    //                    // ✅ 安全处理关键点
    //                    Json::Value kps_arr = f["kps"];
    //                    if (kps_arr.isArray() && kps_arr.size() == 10) {
    //                        for (int i = 0; i < 10; i += 2) {
    //                            FaceKeyPoint kp;
    //                            kp.x = kps_arr[i].asFloat();
    //                            kp.y = kps_arr[i + 1].asFloat();
    //                            ff.kps.push_back(kp);
    //                        }
    //                    }

    //                    // ✅ 处理 embedding 向量
    //                    for (auto& v : f["embedding"])
    //                        ff.embedding.push_back(v.asFloat());

    //                    person_face.faces.push_back(ff);
    //                }

    //                face_features.push_back(person_face);
    //            }
    //            result = true;
    //        }
    //    }

    //    return result;
    //}
    //bool AlgorithmWithApi::parseObjectDetect(std::string& response,
    //    std::vector<AlgorithmDetectObject>& detects) {
    //    std::vector<PersonWithFace> dummy;
    //    return parseObjectDetect(response, detects, dummy);
    //}

bool AlgorithmWithApi::objectDetect(int height, int width, unsigned char* bgr,
    std::vector<AlgorithmDetectObject>& detects,
    std::vector<PersonWithFace>& face_features,
    float& happenScore)
{
    std::string imageBase64;
    analy_compressBgrAndEncodeBase64(height, width, 3, bgr, imageBase64);

    std::string host = mApiHosts[rand() % mApiHosts.size()];
    std::string url = host + "/image/objectDetect";

    Json::Value param;
    param["appKey"] = "s84dsd#7hf34r3jsk@fs$d#$dd";
    param["algorithm"] = mAlgorithmType;
    param["image_base64"] = imageBase64;

    std::string response;
    Request request;
    bool ok = request.post(url.c_str(), param.toStyledString().c_str(), response);
    if (!ok) return false;

    if (mAlgorithmType == "anomaly_autoencoder")
        return parseObjectDetect(response, detects, face_features, happenScore);
    else {
        happenScore = 100.0f;
        return parseObjectDetect(response, detects, face_features);
    }
}

bool AlgorithmWithApi::objectDetect(int height, int width, unsigned char* bgr,
    std::vector<AlgorithmDetectObject>& detects)
{
    std::vector<PersonWithFace> dummy;
    float dummyScore = 100.0f;
    return objectDetect(height, width, bgr, detects, dummy, dummyScore);
}

bool AlgorithmWithApi::objectDetect(int height, int width, unsigned char* bgr,
    std::vector<AlgorithmDetectObject>& detects,
    std::vector<PersonWithFace>& face_features)
{
    float dummyScore = 100.0f;
    return objectDetect(height, width, bgr, detects, face_features, dummyScore);
}

bool AlgorithmWithApi::parseObjectDetect(std::string& response,
    std::vector<AlgorithmDetectObject>& detects,
    std::vector<PersonWithFace>& face_features,
    float& happenScore)
{
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    Json::Value root;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());

    bool result = false;

    bool ok = reader->parse(response.data(), response.data() + response.size(), &root, &errs);
    if (!ok || root["code"].asInt() != 1000) return false;

    Json::Value res = root["result"];
    happenScore = res.get("happen_score", 100.0f).asFloat();

    // 解析检测到的物体
    Json::Value detect_data = res["detect_data"];
    for (auto& i : detect_data) {
        AlgorithmDetectObject object;
        Json::Value loc = i["location"];
        object.x1 = loc["x1"].asInt();
        object.y1 = loc["y1"].asInt();
        object.x2 = loc["x2"].asInt();
        object.y2 = loc["y2"].asInt();
        object.score = i["score"].asFloat();
        object.class_name = i["class_name"].asString();
        detects.push_back(object);
    }

    Json::Value face_json = res["face_features"];
    for (auto& item : face_json) {
        PersonWithFace person_face;
        if (item.isMember("person_id"))
            person_face.person_id = item["person_id"].asInt();
        Json::Value box = item["person_box"];
        person_face.person.x1 = box["x1"].asInt();
        person_face.person.y1 = box["y1"].asInt();
        person_face.person.x2 = box["x2"].asInt();
        person_face.person.y2 = box["y2"].asInt();

        Json::Value face_list = item["faces"];
        for (auto& f : face_list) {
            FaceFeature ff;

            // 填充 bbox
            for (auto& v : f["bbox"])
                ff.bbox.push_back(v.asInt());

            // 解析关键点
            Json::Value kps = f["kps"];
            if (kps.isArray() && kps.size() % 2 == 0) {
                for (int i = 0; i < kps.size(); i += 2) {
                    if (kps[i].isNumeric() && kps[i + 1].isNumeric()) {
                        FaceKeyPoint kp;
                        kp.x = kps[i].asFloat();
                        kp.y = kps[i + 1].asFloat();
                        ff.kps.push_back(kp);
                    }
                }
            }
            // 解析 embedding
            for (auto& v : f["embedding"])
                ff.embedding.push_back(v.asFloat());
            person_face.faces.push_back(ff);
        }
        face_features.push_back(person_face);
    }
    return true;
}

bool AlgorithmWithApi::parseObjectDetect(std::string& response,
    std::vector<AlgorithmDetectObject>& detects,
    std::vector<PersonWithFace>& face_features)
{
    float dummyScore = 100.0f;
    return parseObjectDetect(response, detects, face_features, dummyScore);
}

    //Analyzer::Analyzer(Scheduler* scheduler, Control* control) :
    //    mScheduler(scheduler),
    //    mControl(control)
    //{
    //    mAlgorithm = new AlgorithmWithApi(scheduler->getConfig());
    //}
    Analyzer::Analyzer(Scheduler* scheduler, Control* control)
        : mScheduler(scheduler), mControl(control)
    {
        mAlgorithm = new AlgorithmWithApi(
            mScheduler->getConfig()->algorithmApiHosts,
            mapBehaviorToAlgorithm(mControl->behaviorCode)
        );
    }

    Analyzer::~Analyzer()
    {
        if (mAlgorithm) {
            delete mAlgorithm;
            mAlgorithm = nullptr;
        }
        mDetects.clear();

    }

    bool Analyzer::checkVideoFrame(bool check, int64_t frameCount, unsigned char* data, float& happenScore) {
        happenScore = 0.0f;
        bool happen = false;
        float threshold = 100.0f;
        // 构造图像
        cv::Mat image(mControl->videoHeight, mControl->videoWidth, CV_8UC3, data);

        // 调用接口获取检测信息
        //std::vector<PersonWithFace> face_features;
        mDetects.clear();
        mFeatures.clear();
        mAbnormal.clear();
        mAlgorithm->objectDetect(mControl->videoHeight, mControl->videoWidth, data, mDetects, mFeatures, happenScore);

        std::string algorithmType = mAlgorithm->getAlgorithmType();
        bool isAbnormal = false;

        // ✴️ 区分不同算法的处理方式
        if (algorithmType == "anomaly_autoencoder") {
            isAbnormal = happenScore < threshold;

            AbnormalDetectInfo info;
            info.happenScore = happenScore;
            info.threshold = threshold;
            info.isAbnormal = isAbnormal;
            mAbnormal.push_back(info);

            // 始终显示异常分数
            std::string scoreText = "abnormal score: " + std::to_string(happenScore);
            cv::putText(image, scoreText, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

            // 若异常则居中显示 warning
            if (happenScore <= 10.0f) {
                std::string anomalyText = "warning";
                int fontFace = cv::FONT_HERSHEY_DUPLEX;
                double fontScale = 1.2;
                int thickness = 2;
                int baseline = 0;

                cv::Size textSize = cv::getTextSize(anomalyText, fontFace, fontScale, thickness, &baseline);
                cv::Point origin(mControl->videoWidth / 2 - textSize.width / 2, 120);  // 居中偏下

                cv::putText(image, anomalyText, origin, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
            }
        }


        else if (algorithmType == "face_recognition") {
            // 绘制人物框
            for (const auto& d : mDetects) {
                cv::rectangle(image, cv::Rect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1), cv::Scalar(0, 255, 0), 2);
                std::string label = d.class_name + "-" + std::to_string(d.score);
                cv::putText(image, label, cv::Point(d.x1, d.y1 + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }

            // 绘制人脸框和关键点（不显示ID）
            for (const auto& person_face : mFeatures) {
                for (const auto& face : person_face.faces) {
                    if (face.bbox.size() == 4) {
                        int fx1 = face.bbox[0], fy1 = face.bbox[1];
                        int fx2 = face.bbox[2], fy2 = face.bbox[3];
                        cv::rectangle(image, cv::Rect(fx1, fy1, fx2 - fx1, fy2 - fy1), cv::Scalar(255, 0, 0), 2);
                    }

                    // 绘制关键点（红色）
                    for (const auto& kp : face.kps) {
                        cv::circle(image, cv::Point(kp.x, kp.y), 2, cv::Scalar(0, 0, 255), -1);
                    }
                }
            }
        }

        // 公共信息
        std::string info = "checkFps:" + std::to_string(mControl->checkFps);
        cv::putText(image, info, cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

        return isAbnormal;
    }

    bool Analyzer::checkAudioFrame(bool check, int64_t frameCount, unsigned char* data, int size) {

        return false;
    }
    const std::vector<AlgorithmDetectObject>& Analyzer::getDetects() const {
        return mDetects;
    }
    const std::vector<PersonWithFace>& Analyzer::getFeatures() const {
        return mFeatures;
    }
    const AlgorithmWithApi* Analyzer::getAlgorithm() const {
        return mAlgorithm;
    }
}