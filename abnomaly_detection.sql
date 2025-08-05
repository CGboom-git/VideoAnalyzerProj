/*
 Navicat Premium Data Transfer

 Source Server         : mysql
 Source Server Type    : MySQL
 Source Server Version : 80042
 Source Host           : localhost:3306
 Source Schema         : videoanalyzer

 Target Server Type    : MySQL
 Target Server Version : 80042
 File Encoding         : 65001

 Date: 25/07/2025 17:05:10
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for abnomaly_detection
-- ----------------------------
DROP TABLE IF EXISTS `abnomaly_detection`;
CREATE TABLE `abnomaly_detection`  (
  `abnormaly_id` int NOT NULL AUTO_INCREMENT,
  `frame_id` int NOT NULL,
  `happen_score` float NULL DEFAULT NULL,
  `isAbnormaly` int NULL DEFAULT NULL,
  `threshold` float NULL DEFAULT NULL,
  PRIMARY KEY (`abnormaly_id`, `frame_id`) USING BTREE,
  INDEX `5`(`frame_id` ASC) USING BTREE,
  CONSTRAINT `5` FOREIGN KEY (`frame_id`) REFERENCES `video_frame_info` (`frame_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of abnomaly_detection
-- ----------------------------

SET FOREIGN_KEY_CHECKS = 1;
