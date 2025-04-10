#!/bin/bash

# 遍历当前目录下的所有子目录
for dir in */; do
    # 检查子目录中是否存在 create.sh 文件
    if [ -f "${dir}create.sh" ]; then
        echo "Entering directory: $dir"
        # 进入子目录并执行 create.sh
        (cd "$dir" && bash create.sh)
    else
        echo "No create.sh found in $dir"
    fi
done

