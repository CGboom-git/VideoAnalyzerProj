// MySQLManager.h
#ifndef MYSQLMANAGER_H
#define MYSQLMANAGER_H

#include <string>
#include <memory>
#include <mutex>

namespace sql {
    class Connection;
    class PreparedStatement;
    class Driver;
}

namespace AVSAnalyzer {

    class MySQLManager {
    public:
        static MySQLManager& getInstance();  // 

        void setDatabase(const std::string& host, const std::string& user,
            const std::string& password, const std::string& database, int port = 3306);

        bool connect();
        void disconnect();

        bool insertVideoInfo(const std::string& cameraId,
            int pktSize, int codecId);
        int insertFrameInfo(const int videoInfoId,
            int frameIndex, int width, int height,
            int channels, int codecId);
        int insertTargetInfo(int frameId, int x, int y, int w, int h,
            const std::string& className, float score);
        int insertPersonInfo( int id,
            int x, int y, int w, int h,
            const std::string& embedding,
            const std::string& keypoints
        );
        bool insertAbnormalDetection(
            int frameId,
            float happenScore,
            float threshold,
            bool isAbnormal
        );
        int getLastInsertedTargetId();
    private:
        MySQLManager();
        ~MySQLManager();

        MySQLManager(const MySQLManager&) = delete;
        MySQLManager& operator=(const MySQLManager&) = delete;

        std::string host, user, password, database;
        int port;

        sql::Driver* driver;
        std::unique_ptr<sql::Connection> conn;
        std::mutex conn_mutex;
    };

} // namespace AVSAnalyzer

#endif // MYSQLMANAGER_H
