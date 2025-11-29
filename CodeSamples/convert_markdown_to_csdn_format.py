import re
import sys

def convert_markdown_to_csdn_format(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		content = f.read()

	# 替换 \( 后面可能有多个空格，\) 前面可能有多个空格
	content = re.sub(r'\\\(\s*', '$', content)
	content = re.sub(r'\s*\\\)', '$', content)
	# 替换 \[ 和 \]
	content = content.replace('\\[', '$$').replace('\\]', '$$')

	with open(file_path, 'w', encoding='utf-8') as f:
		f.write(content)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Usage: python convert_markdown_to_csdn_format.py <file_path>')
		sys.exit(1)
	convert_markdown_to_csdn_format(sys.argv[1])
