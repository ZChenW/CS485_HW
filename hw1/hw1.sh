#!/bin/bash

# Use Unix commands to print the name of each character with dialogue in the script, one name per line.
echo "Question 1.5"
# TODO: Replace the following line with your code.
# gerp - E => Extended Regular Expression
# ^ (From start) | [[:space:]]{10,} (at least 10 spaces) | [A-Z] (Start from Upper case) | [A-Z0-9 .,'-]* (0 or multiplate times) |
# ([[:space:]]*\\([^)]*\\))? <=> [[:space:]]* (0 or multiplate times) | \\( => \( => ( | \\) => \) => ) | [^)]* (不是右括号的任意字符)*
# [[:space:]]*$" The space at the end
# sed -E => 连续替换
# s/^ +// => 去掉行首的空格缩进 | s/ (replace) | // (replace item) | ^ + (The start of line and space)
# s/ *\([^)]*\)//g => * 零个或多个空格 | \(\) => () | [^)]* => 任意不是 ) 的字符，重复零次或多次 | /g => global
# sort -u 排序 + 去重
grep -E "^[[:space:]]{10,}[A-Z][A-Z0-9 .,'-]*([[:space:]]*\\([^)]*\\))?[[:space:]]*$" lotr_script.txt |
	sed -E 's/^ +//; s/ *\([^)]*\)//g; s/ +$//; s/  +/ /g' |
	sort -u
echo "I Implemented"

# First, extract all lines of dialogue in this script.
# Then, normalize and tokenize this text such that all alphabetic characters are converted to lowercase and words are sequences of alphabetic characers.
# Finally, print the top-20 most frequent word types and their corresponding counts.
# Hint: Ignore parantheticals. These contain short stage directions.
echo
echo "Question 1.6"
# TODO: Replace the following line with your code.
# awk "..." lotr_script.txt => 提取对话行 | d=1 in the txt and d=0 not in the txt
# BEGIN{d=0} => 默认不再对话行
# match($0,/^[[:space:]]+/) && RLENGTH>=10 => 匹配第一行的空白行，必须>=10个空白行
# name = substr($0, RLENGTH+1) => 去掉行首空白
# name ~ ... => 判断 name 是否像“角色名行
# /^[A-Z][A-Z0-9 .-]*(\([^)]*\))?[[:space:]]*$/ => 上面的Regular Expression
# d=1; next => 切换到对话块，跳过名字来到下一行
# d && /^[[:space:]]*$/ { d=0; next } => 如果d=1而且当前行为空白行，d=0退出空白行（在文本中我们可以看出，遇到空白行，也就是结束当前人物对话）
# {} => 对于每一个行，运行这一行
# $0 是 awk 的内置变量：表示当前整行文本，把当前文本行赋值给line
# sub(/^[[:space:]]+/, "", line) => 去掉行首的空白
# if (line ~ /^\(/) next => 如果line 去掉空白之后的结果是()，则跳过当前行 => 比如 (horrified) or (melodramatic)
# print $0 => 输出原行
# tr 'A-Z' 'a-z'  => 大写转换为小写
# sort => 排序
# uniq -c 计数
# sort -nr 按数字排序-降序
# head -20 提取前20个
# python3 -c "import sys; out=[]; for line in sys.stdin:    c,w=line.split()    out.append((w, int(c))) print(out)"
# 转化为[(word_1, count_1), (word_2, count_2), ...]形式
awk '
BEGIN{d=0}
{
  if (match($0,/^[[:space:]]+/) && RLENGTH>=10) {
    name = substr($0, RLENGTH+1)
    if (name ~ /^[A-Z][A-Z0-9 .-]*(\([^)]*\))?[[:space:]]*$/) {
      d=1; next
    }
  }
}
d && /^[[:space:]]*$/ { d=0; next }
d {
  line=$0
  sub(/^[[:space:]]+/, "", line)
  if (line ~ /^\(/) next
  print $0
}
' lotr_script.txt |
	tr 'A-Z' 'a-z' |
	grep -oE '[a-z]+' |
	sort |
	uniq -c |
	sort -nr |
	head -20 |
	python3 -c "import sys; out=[];
for line in sys.stdin:
    c,w=line.split()
    out.append((w, int(c)))
print(out)"
#echo "I Implemented"
