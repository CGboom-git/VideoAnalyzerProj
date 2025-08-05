#ifndef ANALYZER_ANALYZER_H
#define ANALYZER_ANALYZER_H

#include <string>
#include <vector>

namespace AVSAnalyzer {
	struct Control;
	class Config;
	class Scheduler;

	struct AlgorithmDetectObject
	{
		int x1;
		int y1;
		int x2;
		int y2;
		float score;
		std::string class_name;
	};

	struct FaceKeyPoint {
		float x;
		float y;
	};

	struct FaceFeature {
		std::vector<int> bbox;                  // [x1, y1, x2, y2]
		std::vector<FaceKeyPoint> kps;          // 5个关键点
		std::vector<float> embedding;           // 512维向量
	};

	struct PersonWithFace {
		AlgorithmDetectObject person;           // 人物框
		std::vector<FaceFeature> faces;         // 多张人脸
		int person_id = -1;
	};

	struct AbnormalDetectInfo {
		float happenScore = 100;   // 异常得分，默认最大值
		float threshold = 10;     // 判断阈值，默认等于最大值
		bool isAbnormal = false;     // 是否异常
	};

	class AlgorithmWithApi
	{
	public:
		AlgorithmWithApi() = delete;
		//AlgorithmWithApi(Config* config);
		AlgorithmWithApi(const std::vector<std::string>& hosts, const std::string& algorithmType);
		~AlgorithmWithApi();

	public:
		bool test();

		//// 原始YOLO检测接口
		//bool objectDetect(int height, int width, unsigned char* bgr, std::vector<AlgorithmDetectObject>& detects);

		//// 添加重载版本，支持人脸特征提取
		//bool objectDetect(int height, int width, unsigned char* bgr,
		//	std::vector<AlgorithmDetectObject>& detects,
		//	std::vector<PersonWithFace>& face_features);
		const std::string& getAlgorithmType() const { return mAlgorithmType; }

		bool objectDetect(int height, int width, unsigned char* bgr, std::vector<AlgorithmDetectObject>& detects);

		bool objectDetect(int height, int width, unsigned char* bgr,
			std::vector<AlgorithmDetectObject>& detects,
			std::vector<PersonWithFace>& face_features);

		bool objectDetect(int height, int width, unsigned char* bgr,
			std::vector<AlgorithmDetectObject>& detects,
			std::vector<PersonWithFace>& face_features,
			float& happenScore);
		std::string mAlgorithmType;

	private:
		//// 原始JSON解析函数
		//bool parseObjectDetect(std::string& response, std::vector<AlgorithmDetectObject>& detects);

		//// 添加重载版本，支持人脸特征提取
		//bool parseObjectDetect(std::string& response,
		//	std::vector<AlgorithmDetectObject>& detects,
		//	std::vector<PersonWithFace>& face_features);
		bool parseObjectDetect(std::string& response, std::vector<AlgorithmDetectObject>& detects);

		bool parseObjectDetect(std::string& response,
			std::vector<AlgorithmDetectObject>& detects,
			std::vector<PersonWithFace>& face_features);

		bool parseObjectDetect(std::string& response,
			std::vector<AlgorithmDetectObject>& detects,
			std::vector<PersonWithFace>& face_features,
			float& happenScore);
	private:
		Config* mConfig;
		std::vector<std::string> mApiHosts;
	};

	class Analyzer {
	public:
		explicit Analyzer(Scheduler* scheduler, Control* control);
		~Analyzer();

	public:
		bool checkVideoFrame(bool check, int64_t frameCount, unsigned char* data, float& happenScore);
		bool checkAudioFrame(bool check, int64_t frameCount, unsigned char* data, int size);
		const std::vector<AlgorithmDetectObject>& getDetects() const;
		const std::vector<PersonWithFace>& getFeatures() const;
		const std::vector<AbnormalDetectInfo>& getAbnormalInfos() const { return mAbnormal; }
		const AlgorithmWithApi* getAlgorithm() const;
	private:
		Scheduler* mScheduler;
		Control* mControl;
		AlgorithmWithApi* mAlgorithm;
		std::vector<AlgorithmDetectObject> mDetects;
		std::vector<PersonWithFace> mFeatures;
		std::vector<AbnormalDetectInfo> mAbnormal;
	};
}
#endif //ANALYZER_ANALYZER_H

