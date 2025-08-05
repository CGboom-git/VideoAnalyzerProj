// MySQLManager.cpp
#include "MySQLManager.h"
#include <jdbc/mysql_driver.h>
#include <jdbc/mysql_connection.h>
#include <jdbc/cppconn/prepared_statement.h>
#include <jdbc/cppconn/resultset.h>
#include <iostream>

namespace AVSAnalyzer {

    MySQLManager::MySQLManager()
        : driver(nullptr), port(3306) {
    }

    MySQLManager::~MySQLManager() {
        disconnect();
    }

    MySQLManager& MySQLManager::getInstance() {
        static MySQLManager instance;
        return instance;
    }

    void MySQLManager::setDatabase(const std::string& host_,
        const std::string& user_,
        const std::string& password_,
        const std::string& database_,
        int port_) {
        host = host_;
        user = user_;
        password = password_;
        database = database_;
        port = port_;
    }

    bool MySQLManager::connect() {
        std::lock_guard<std::mutex> lock(conn_mutex);
        try {
            driver = sql::mysql::get_mysql_driver_instance();
            std::string url = "tcp://" + host + ":" + std::to_string(port);
            conn.reset(driver->connect(url, user, password));
            conn->setSchema(database);
            return true;
        }
        catch (sql::SQLException& e) {
            std::cerr << "[MySQLManager] Connection failed: " << e.what() << std::endl;
            return false;
        }
    }

    void MySQLManager::disconnect() {
        std::lock_guard<std::mutex> lock(conn_mutex);
        if (conn) {
            conn->close();
            conn.reset();
        }
    }

    bool MySQLManager::insertVideoInfo(const std::string& cameraId,
        int pktSize, int codecId) {
        std::lock_guard<std::mutex> lock(conn_mutex);
        try {
            if (!conn) {
                if (!connect()) return false;
            }

            std::unique_ptr<sql::PreparedStatement> stmt(
                conn->prepareStatement("INSERT INTO video_info (camera_id, pkt_size_bytes, codec_id) VALUES ( ?, ?, ?)"));
            stmt->setString(1, cameraId);
            stmt->setInt(2, pktSize);
            stmt->setInt(3, codecId);
            stmt->execute();
            std::unique_ptr<sql::Statement> stmt2(conn->createStatement());
            std::unique_ptr<sql::ResultSet> res(stmt2->executeQuery("SELECT LAST_INSERT_ID()"));
            if (res->next()) {
                return res->getInt(1);  // 返回新插入 video_info 的主键
            }
            return true;
        }
        catch (sql::SQLException& e) {
            std::cerr << "[MySQLManager] Insert failed: " << e.what() << std::endl;
            return false;
        }
    }

    int MySQLManager::insertFrameInfo(const int videoInfoId,
        int frameIndex,
        int width,
        int height,
        int channels,
        int codecId) {
        std::lock_guard<std::mutex> lock(conn_mutex);
        try {
            if (!conn && !connect()) {
                std::cerr << "[MySQLManager] insertFrameInfo failed: DB not connected." << std::endl;
                return -1;
            }

            std::unique_ptr<sql::PreparedStatement> stmt(
                conn->prepareStatement(
                    "INSERT INTO video_frame_info (video_id, frame_index, width, height, channels, codec_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                )
            );

            stmt->setInt(1, videoInfoId);
            stmt->setInt(2, frameIndex);
            stmt->setInt(3, width);
            stmt->setInt(4, height);
            stmt->setInt(5, channels);
            stmt->setInt(6, codecId);

            stmt->execute();
            std::unique_ptr<sql::Statement> stmt2(conn->createStatement());
            std::unique_ptr<sql::ResultSet> res(stmt2->executeQuery("SELECT LAST_INSERT_ID()"));
            if (res->next()) {
                return res->getInt(1); // 这就是 frame_id
            }
            return 1;  // 成功返回 1

        }
        catch (const sql::SQLException& e) {
            std::cerr << "[MySQLManager] insertFrameInfo SQL error: " << e.what() << std::endl;
            return -1;
        }
    }

    int MySQLManager::insertTargetInfo(int frameId, int x, int y, int w, int h,
        const std::string& className, float score) {
        std::lock_guard<std::mutex> lock(conn_mutex);
        try {
            if (!conn && !connect()) return -1;

            std::unique_ptr<sql::PreparedStatement> stmt(
                conn->prepareStatement(
                    "INSERT INTO detection_result (frame_id, x, y, w, h, class, score) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)"
                )
            );

            stmt->setInt(1, frameId);
            stmt->setInt(2, x);
            stmt->setInt(3, y);
            stmt->setInt(4, w);
            stmt->setInt(5, h);
            stmt->setString(6, className);
            stmt->setDouble(7, score);
            stmt->execute();
            //std::unique_ptr<sql::Statement> stmt2(conn->createStatement());
            //std::unique_ptr<sql::ResultSet> res(stmt2->executeQuery("SELECT LAST_INSERT_ID()"));
            //if (res->next()) {
            //    return res->getInt(1); // 这就是 frame_id
            //}
            return 1;
        }
        catch (const sql::SQLException& e) {
            std::cerr << "[MySQLManager] insertTargetInfo failed: " << e.what() << std::endl;
            return -1;
        }
    }

    int MySQLManager::getLastInsertedTargetId() {
        std::lock_guard<std::mutex> lock(conn_mutex);  // 确保线程安全
        try {
            if (!conn && !connect()) return -1;

            // 查询获取最后插入的 target_id
            std::unique_ptr<sql::Statement> stmt(conn->createStatement());
            std::unique_ptr<sql::ResultSet> res(stmt->executeQuery(
                "SELECT target_id FROM detection_result ORDER BY target_id DESC LIMIT 1"
            ));

            if (res->next()) {
                return res->getInt("target_id"); // 获取最后插入的 target_id
            }
            return -1;  // 如果没有数据，返回 -1
        }
        catch (const sql::SQLException& e) {
            std::cerr << "[MySQLManager] getLastInsertedTargetId failed: " << e.what() << std::endl;
            return -1;
        }
    }

    int MySQLManager::insertPersonInfo(
        int id,
        int x, int y, int w, int h,
        const std::string& embedding,
        const std::string& keypoints)
    {
        try {
            std::unique_ptr<sql::PreparedStatement> pstmt(conn->prepareStatement(
                "INSERT INTO person_info "
                "(target_id,face_bbox_x, face_bbox_y, face_bbox_h, face_bbox_w, embedding_feature, keypoint) "
                "VALUES (?,?, ?, ?, ?, ?, ?)"
            ));
            pstmt->setInt(1, id);
            pstmt->setInt(2, x);
            pstmt->setInt(3, y);
            pstmt->setInt(4, h);
            pstmt->setInt(5, w);
            pstmt->setString(6, embedding);
            pstmt->setString(7, keypoints);

            pstmt->executeUpdate();
            return 1; // 插入成功返回1
        }
        catch (sql::SQLException& e) {
            std::cerr << "[MySQLManager] insertPersonInfo failed: " << e.what() << std::endl;
            return -1;
        }
    }

    bool MySQLManager::insertAbnormalDetection(
        int frameId,
        float happenScore,
        float threshold,
        bool isAbnormal
    ) {
        std::lock_guard<std::mutex> lock(conn_mutex);
        try {
            if (!conn && !connect()) return false;

            std::unique_ptr<sql::PreparedStatement> pstmt(
                conn->prepareStatement(
                    "INSERT INTO abnomaly_detection (frame_id, happen_score, threshold, isAbnormaly) "
                    "VALUES (?, ?, ?, ?)"
                )
            );

            pstmt->setInt(1, frameId);
            pstmt->setDouble(2, happenScore);
            pstmt->setDouble(3, threshold);
            pstmt->setBoolean(4, isAbnormal);

            pstmt->executeUpdate();
            return true;
        }
        catch (const sql::SQLException& e) {
            std::cerr << "[MySQLManager] insertAbnormalDetection failed: " << e.what() << std::endl;
            return false;
        }
    }

} // namespace AVSAnalyzer
