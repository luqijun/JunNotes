#!/bin/bash

set -e  # 遇错退出

# ====== 日志函数（写到 stdout/stderr，同时可写到日志文件） ======
_log_write() {
  # args: level, message
  if [[ -n "$LOG_FILE" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    printf '%s %s: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" >>"$LOG_FILE"
  fi
}

log_info() {
  echo "INFO: $1"
  _log_write "INFO" "$1"
}

log_warn() {
  echo "WARN: $1" >&2
  _log_write "WARN" "$1"
}

log_error() {
  echo "ERROR: $1" >&2
  _log_write "ERROR" "$1"
}

# ====== 帮助信息 ======
show_help() {
  cat <<EOF
用法: $0 [-l|--log <log_file>] <start_commit> [end_commit] [output_dir]

功能:
  提取指定 Git 提交范围内所有提交所修改过的文件（去重），
  并将这些文件在 end_commit 时的历史版本复制到 output_dir（保留目录结构）。

参数:
  -l, --log      可选，指定日志文件路径。如果未指定，默认写入到输出目录下的 extract.log
  start_commit   起始提交（包含）
  end_commit     结束提交（包含），可选；若省略，默认为 HEAD
  output_dir     输出目录，可选；默认为 extracted_files_<range>

示例:
  $0 abc1234                    # 从 abc1234 到 HEAD，日志写到默认位置
  $0 -l /tmp/out.log abc1234    # 指定日志文件
  $0 abc1234 def5678 my_output  # 指定输出目录

EOF
}

# ====== 参数解析 ======
# ====== 可选开关解析（仅 -l/--log） ======
LOG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -l|--log)
      shift
      LOG_FILE="$1"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      log_error "未知选项: $1"
      show_help
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  log_error "至少需要一个提交哈希或引用。"
  show_help
  exit 1
fi

START_COMMIT="$1"
END_COMMIT="${2:-HEAD}"
OUTPUT_DIR="${3:-}"

# 验证 START_COMMIT 是否有效
if ! git rev-parse --verify --quiet "$START_COMMIT" >/dev/null; then
  log_error "起始提交 '$START_COMMIT' 无效或不存在。"
  exit 1
fi

# 验证 END_COMMIT 是否有效
if ! git rev-parse --verify --quiet "$END_COMMIT" >/dev/null; then
  log_error "结束提交 '$END_COMMIT' 无效或不存在。"
  exit 1
fi

# 获取完整的 commit hash（用于构造范围和输出目录）
START_FULL=$(git rev-parse "$START_COMMIT")
END_FULL=$(git rev-parse "$END_COMMIT")

# 构造 log 范围：包含 START_COMMIT 本身
RANGE="$START_FULL^..$END_FULL"

# 如果 START_COMMIT 是仓库的第一个提交（没有父提交），则不能用 ^，改用单点
if ! git rev-parse --verify --quiet "$START_FULL^" >/dev/null; then
  log_warn "起始提交是初始提交，无法使用 ^，将使用 $START_FULL..$END_FULL（可能不包含起始提交的变更）"
  RANGE="$START_FULL..$END_FULL"
fi

# 默认输出目录
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="extracted_files_${START_FULL:0:8}_${END_FULL:0:8}"
fi

# 如果用户未指定日志文件，默认写到输出目录下的 extract.log（在输出目录确定后设置）
if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$OUTPUT_DIR/extract.log"
fi

# 初始化日志（覆盖旧日志，确保所在目录存在）
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
: >"$LOG_FILE"

# ====== 获取变更文件列表 ======
log_info "正在分析提交范围: $RANGE"
TEMP_FILE=$(mktemp)
# 使用 git log 获取范围内每个提交涉及的文件（去重）
git log --name-only --pretty=format: "$RANGE" | sort -u > "$TEMP_FILE"

# 过滤掉空行
sed -i '/^\s*$/d' "$TEMP_FILE"

FILE_COUNT=$(wc -l < "$TEMP_FILE")
if [[ $FILE_COUNT -eq 0 ]]; then
  log_warn "在指定范围内未找到任何变更文件。"
  rm -f "$TEMP_FILE"
  exit 0
fi

log_info "找到 $FILE_COUNT 个唯一文件。"

# ====== 创建输出目录 ======
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
log_info "输出目录: $OUTPUT_DIR"

# ====== 复制文件（使用 END_COMMIT 时的历史版本）======
log_info "正在复制文件（基于提交 $END_FULL 的快照）..."

while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    # 跳过被删除的文件（在 END_COMMIT 中不存在）
    if git ls-tree -r --name-only "$END_FULL" | grep -Fqx "$file" 2>/dev/null; then
      mkdir -p "$OUTPUT_DIR/$(dirname "$file")"
      git show "$END_FULL:$file" > "$OUTPUT_DIR/$file"
    else
      log_warn "文件在 $END_FULL 中不存在（可能已被删除）: $file"
    fi
  fi
done < "$TEMP_FILE"

# ====== 清理 & 完成 ======
rm -f "$TEMP_FILE"
log_info "完成！文件已保存至: $(realpath "$OUTPUT_DIR")"