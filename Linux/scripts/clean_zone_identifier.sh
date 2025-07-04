#!/bin/bash
# filepath: /home/lqjun/Projects/Python/CrowdCounting/PET_Origin/clean_zone_identifier.sh

# 检查是否传入文件夹路径
if [ -z "$1" ]; then
  echo "用法: $0 <文件夹路径>"
  exit 1
fi

# 获取文件夹路径
TARGET_DIR="$1"

# 检查目标路径是否为有效目录
if [ ! -d "$TARGET_DIR" ]; then
  echo "错误: $TARGET_DIR 不是有效的目录。"
  exit 1
fi

# 查找并删除所有以 :Zone.Identifier 结尾的文件
find "$TARGET_DIR" -type f -name '*:Zone.Identifier' -exec rm -f {} \;

echo "已清除 $TARGET_DIR 下所有以 :Zone.Identifier 结尾的文件。"
