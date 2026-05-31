import re
import random


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


def generate_industrial_phrases(positions, sub_positions, parts, lang="zh", count=10000):
    phrases = set()
    while len(phrases) < count:
        pos = random.choice(positions)
        sub = random.choice(sub_positions)
        part = random.choice(parts)

        # 随机组合逻辑
        variant = random.random()
        if variant > 0.6:
            phrase = f"{pos}{sub}{part}" if lang == "zh" else f"{pos} {part} {sub}"
        elif variant > 0.3:
            phrase = f"{pos}{part}{sub}" if lang == "zh" else f"{pos} {part} {sub}"
        else:
            phrase = f"{sub}{pos}{part}" if lang == "zh" else f"{pos} {part} {sub}"

        phrases.add(phrase)
    return list(phrases)

if __name__ == "__main__":

    # 生成并保存
    # 中文示例
    # 定义零件的核心组件及方位属性
    positions = ["前", "后", "左前", "右前", "左后", "右后", "前排", "后排", "顶棚", "顶部", "底部"]
    sub_positions = ["上", "下", "内", "外", "左", "右"]
    parts = read_and_split("industrial_part_names_zh.txt")
    data_zh = generate_industrial_phrases(positions, sub_positions, parts, lang="zh", count=20000)
    # 英文示例
    positions = ["front", "rear", "left front", "right front", "left rear",
                "right_rear", "front_row", "rear_row", "roof", "top", "bottom"]
    sub_positions = ["upper", "lower", "inner", "outer", "left", "right"]
    parts = read_and_split("industrial_part_names_en.txt")
    data_en = generate_industrial_phrases(positions, sub_positions, parts, lang="en", count=20000)
    with open("calibration_samples.txt", "w", encoding="utf-8") as f:
        for item in data_zh + data_en:
            f.write(item + "\n")

    print(f"成功生成 {len(data_zh + data_en)} 条校准数据，已保存至 calibration_samples.txt")
