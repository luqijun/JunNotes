import argparse
from modelscope import snapshot_download

def parse_args():
    """解析命令行参数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='通用模型下载脚本')

    # 添加模型名称参数（必填）
    parser.add_argument('--model', type=str, required=True, help='要下载的模型名称，例如：Jerry0/text2vec-base-chinese')

    # 添加保存路径参数（可选，默认值为./models）
    parser.add_argument('--cache_dir', type=str, default='./models', help='模型保存路径，默认：./models')

    # 解析参数
    args = parser.parse_args()
    return args

def main():
    # 下载模型
    args = parse_args()
    model_dir = snapshot_download(args.model, cache_dir=args.cache_dir)

    # 打印下载结果
    print(f"模型已下载至: {model_dir}") # ./models/Jerry0/text2vec-base-chinese

if __name__ == "__main__":
    # 调用主函数
    main()