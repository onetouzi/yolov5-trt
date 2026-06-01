/*
 * @Author: Touzi
 * @Email: wwwtouzi@outlook.com
 * @Description: 读取目录文件列表、读取标签文件等工具函数
 * @Date: 2026-05-30 12:05:20
 */

#pragma once

#include <dirent.h>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names)
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr)
    {
        return -1;
    }

    struct dirent *p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr)
    {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0)
        {
            // std::string cur_file_name(p_dir_name);
            // cur_file_name += "/";
            // cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

// 去除前后空格
static inline std::string trim_leading_whitespace(const std::string &str)
{
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

// 浮点数转化为字符串
static inline std::string to_string_with_precision(const float a_value, const int n = 2)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// 读取标签文件 0：cat 1:dog......
static inline int read_labels(const std::string labels_filename, std::unordered_map<int, std::string> &labels_map)
{

    std::ifstream file(labels_filename);
    // Read each line of the file
    std::string line;
    int index = 0;
    while (std::getline(file, line))
    {
        // Strip the line of any leading or trailing whitespace
        line = trim_leading_whitespace(line);

        // Add the stripped line to the labels_map, using the loop index as the key
        labels_map[index] = line;
        index++;
    }
    // Close the file
    file.close();

    return 0;
}
