import random
import re


def read_and_split(filename, delimiters=',|\n'):
    """
    读取txt文件，以指定分隔符分隔内容，并trim每个元素

    Args:
        filename: 文件路径
        delimiters: 分隔符（正则表达式格式），默认为逗号或换行

    Returns:
        list: 分隔并trim后的列表
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则表达式分隔并trim
        result = [item.strip() for item in re.split(
            delimiters, content) if item.strip()]

        return result

    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []


if __name__ == "__main__":

    # 生成并保存
    # 中文示例
    # 定义零件的核心组件及方位属性
    positions = ["前", "后", "左前", "右前", "左后", "右后", "前排", "后排", "顶棚", "顶部", "底部"]
    sub_positions = ["上", "下", "内", "外", "左", "右"]
    parts = read_and_split("industrial_part_names_zh.txt")
    data_zh = positions + sub_positions + parts
    # 英文示例
    positions = ["front", "rear", "left front", "right front", "left rear",
                 "right_rear", "front_row", "rear_row", "roof", "top", "bottom"]
    sub_positions = ["upper", "lower", "inner", "outer", "left", "right"]
    parts = read_and_split("industrial_part_names_en.txt")
    data_en = positions + sub_positions + parts

    numbers = [str(i) for i in range(1, 101)]
    chars = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

    save_path = "calibration_samples2.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for item in data_zh + data_en + numbers + chars:
            f.write(item + "\n")

    print(f"成功生成 {len(data_zh + data_en)} 条校准数据，已保存至 {save_path}")
