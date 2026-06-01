/*
 * @Author: Touzi
 * @Email: wwwtouzi@outlook.com
 * @Description: yolo核心数据结构
 * @Date: 2026-05-30 11:52:14
 */
#pragma once

#include "config.h"

// 检测核
struct YoloKernel
{
    int width;
    int height;
    float anchors[kNumAnchor * 2];
};

// 检测结果载体
struct alignas(float) Detection
{
    float bbox[4];  // 检测结果边框坐标
    float conf;     // 置信度得分
    float class_id; // 预测类别
    float mask[32]; // 分割掩码，为分割预留的空间
};