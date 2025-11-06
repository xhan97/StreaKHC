#!/bin/bash
#
# 对比测试原始 grinch.py 和新的 grinch_new.py
#

cd "$(dirname "$0")/.."

echo "运行 Grinch 版本对比测试..."
echo "======================================"
echo ""

conda run -n shc python compare_grinch_versions.py 2>&1 | \
    grep -v "^I[0-9]" | \
    grep -v "grinch_build" | \
    grep -v "write file:"

echo ""
echo "======================================"
echo "测试完成！"
echo ""
echo "生成的文件："
echo "  - test_old.tree (原始版本的树结构)"
echo "  - test_new.tree (新版本的树结构)"
