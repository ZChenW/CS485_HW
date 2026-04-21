# Hugging Face 数据集 `Pclanglais/Brahe-Novels` 分析报告

本文基于本地 Hugging Face 数据集目录直接复现分析，路径为：

`/home/chakew/Downloads/spring_2026/CS485_HW/data/brahe_novels/train`

数据通过 `datasets.load_from_disk('.')` 加载。

## 1. 数据集字段和基本结构

### 直接观察

- 数据类型：Hugging Face `Dataset`
- 样本数：`8226`
- 字段：
  - `instruction_id: string`
  - `full_text: string`
  - `analysis: string`

加载结果：

```python
from datasets import load_from_disk
ds = load_from_disk(".")
print(ds)
print(ds.features)
```

对应输出：

```python
Dataset({
    features: ['instruction_id', 'full_text', 'analysis'],
    num_rows: 8226
})
{'instruction_id': Value('string'), 'full_text': Value('string'), 'analysis': Value('string')}
```

### `analysis` 的整体结构

### 直接观察

- `analysis` 不是自由长文本，而是高度规则化的多行字符串。
- 每一行都符合 `Label: value` 形式。
- 全量 `8226` 条中：
  - `0` 条整段无法解析
  - `0` 条未匹配到任何标签
  - `0` 条标签重复
  - `0` 条多行续写字段

也就是说，从“行级语法”上看，`analysis` 可解析性很高。

一个典型样本如下：

```text
Summary: The story is about an old widow and her sons, particularly one son named Vasya who breeds pigeons and sells them for extra money.
Enunciation: Third-person narrative
Tone: Neutral
Genre: Realistic fiction
Speech standard: Conversational
Literary form: Description of a character and their background
Active character: Matvey Savitch, Marfa Semyonovna Kapluntsev, Vasya
Fuzzy time: Ten years ago
Fuzzy place: Little house, tallow and oil factory
```

## 2. `analysis` 中标签集合与频率

### 直接观察

全数据一共观察到 `19` 种标签：

- `Summary` 8223
- `Tone` 8214
- `Speech standard` 8201
- `Literary form` 8163
- `Enunciation` 8148
- `Genre` 7837
- `Active character` 7009
- `Fuzzy place` 5513
- `Narrative arc` 5150
- `Literary movement` 3426
- `Time setting` 3365
- `Intertextuality` 3207
- `Trope` 3190
- `Absolute place` 2228
- `Fuzzy time` 1534
- `Diegetic time` 1449
- `Quoted character` 941
- `Quoted work` 492
- `Absolute time` 475

### 重点标签频率

| 标签 | 出现数 | 占比 |
|---|---:|---:|
| Summary | 8223 | 99.96% |
| Enunciation | 8148 | 99.05% |
| Tone | 8214 | 99.85% |
| Genre | 7837 | 95.27% |
| Speech standard | 8201 | 99.70% |
| Literary form | 8163 | 99.23% |
| Active character | 7009 | 85.21% |
| Fuzzy time | 1534 | 18.65% |
| Fuzzy place | 5513 | 67.02% |

### 直接观察

- `Summary / Tone / Speech standard / Literary form / Enunciation` 几乎总是出现。
- `Genre` 略有缺失，但整体覆盖率仍高。
- `Active character` 不算完全稳定。
- `Fuzzy time` 非常稀疏，只占约 `18.65%`。
- `Fuzzy place` 出现较多，但并非必有。

## 3. 是否采用统一模板

### 直接观察

不是单一固定模板，而是“统一语法 + 多种模板变体”。

- `analysis` 的共同点非常稳定：都采用 `Label: value` 的逐行键值结构。
- 但标签顺序和标签组合并不唯一。
- 全量数据共发现 `1391` 种标签序列。

最常见标签顺序样式包括：

1. `Summary -> Narrative arc -> Enunciation -> Tone -> Genre -> Speech standard -> Literary form -> Active character -> Fuzzy place`（208 条）
2. `Summary -> Enunciation -> Tone -> Genre -> Speech standard -> Literary form -> Active character -> Fuzzy place`（172 条）
3. `Summary -> Trope -> Narrative arc -> Enunciation -> Tone -> Genre -> Speech standard -> Literary form -> Active character -> Fuzzy place`（142 条）

### 直接观察

- 仅 `1331` 条同时包含用户列出的 9 个重点标签。
- 若不要求 `Fuzzy time`，则有 `4891` 条包含其余 8 个重点标签。

### 基于模式的推断

可以把该数据理解为一个“近似模板族”，而不是严格模板。适合做结构化抽取，但不应假设所有样本字段齐全。

## 4. 各重点标签的代表性样本、值分布与不规则性

## 4.1 Summary

### 直接观察

- 出现数：`8223`
- 唯一值数：`8222`
- 说明：`Summary` 基本是样本级自由描述，重复极少。

代表样本：

- `idx 0`: `The story is about an old widow and her sons, particularly one son named Vasya who breeds pigeons and sells them for extra money.`
- `idx 1`: `The text is about the thoughts and feelings of Melrosada towards Isabel, who he believes is a married woman.`
- `idx 2`: `D'Artagnan and Planchet set out on an adventure, confident in their abilities and unbothered by others' opinions.`

缺失样本：

- `idx 2121`
- `idx 2787`
- `idx 3619`

## 4.2 Enunciation

### 直接观察

- 出现数：`8148`
- 唯一值数：`379`
- 高频值：
  - `Third-person narrative` 4231
  - `First-person narrative` 1910
  - `Dialog` 772
  - `Dialogue` 468
  - `Multiple characters speaking in dialogue` 81
  - `Third-person narrator` 62

代表样本：

- `idx 0`: `Third-person narrative`
- `idx 3`: `First-person narrative`
- `idx 20`: `Dialog`
- `idx 40`: `Dialog between characters`

### 直接观察

存在明显变体：

- `Dialog`
- `Dialogue`
- `Dialogue between characters`
- `Multiple characters speaking in dialogue`
- `Third-person narrative with dialogue`
- `Third-person narrator`

还有大量非标准写法，例如直接写说话者：

- `D. Carlos, Izabel, D. Emilio, D. Ramon`
- `Mr. Wegg and Mr. Boffin are speaking in the text`
- `Jesus speaking to his disciples`

### 基于模式的推断

`Enunciation` 可以归一化为以下几类：

- `third_person`
- `first_person`
- `dialogue`
- `mixed`
- `other`

一个粗略归并结果为：

- 第三人称：`4313`
- 第一人称：`1942`
- 对话类：`1605`
- 混合类：`40`
- 其他自由写法：`248`

## 4.3 Tone

### 直接观察

- 出现数：`8214`
- 唯一值数：`1811`
- 高频值：
  - `Scholarly` 883
  - `Tragic` 817
  - `Serious` 467
  - `Humorous` 239
  - `Informative` 227
  - `Reflective` 188

代表样本：

- `idx 1`: `Melancholic`
- `idx 4`: `Envious, reflective, curious`
- `idx 492`: `Playful/sarcastic`

### 直接观察

- 存在单值、逗号并列、多标签斜杠写法。
- 例如：
  - `Serious, tense`
  - `Envious, reflective, curious`
  - `Anxious/frustrated`
  - `Playful/sarcastic`

### 基于模式的推断

`Tone` 适合拆成字符串列表，但不适合在不人工制定词表的前提下强行映射到小规模闭集。

## 4.4 Genre

### 直接观察

- 出现数：`7837`
- 唯一值数：`1116`
- 高频值：
  - `Historical fiction` 638
  - `Drama` 638
  - `Romance` 385
  - `Historical novel` 344
  - `Adventure` 330
  - `Poetry` 316

代表样本：

- `idx 0`: `Realistic fiction`
- `idx 5`: `Mystery/Crime Fiction`
- `idx 39`: `Non-fiction, historical account`

### 直接观察

变体明显：

- 单一体裁：`Adventure`
- 复合体裁：`Drama, Crime Fiction`
- 斜杠变体：`Mystery/Crime Fiction`
- 近义重叠：`Historical fiction` 与 `Historical novel`

## 4.5 Speech standard

### 直接观察

- 出现数：`8201`
- 唯一值数：`485`
- 高频值：
  - `Standard` 1227
  - `Poetic` 923
  - `Informal` 791
  - `Formal` 652
  - `Conversational` 644

代表样本：

- `idx 0`: `Conversational`
- `idx 1`: `Standard language`
- `idx 2`: `Standard`
- `idx 49`: `Informal, colloquial`

### 直接观察

存在若干近义变体：

- `Standard`
- `Standard language`
- `Standard English`
- `Standard literary language`
- `Standard literary`

以及并列值：

- `Informal, colloquial`
- `Poetic, elevated language`
- `Formal, elevated`

## 4.6 Literary form

### 直接观察

- 出现数：`8163`
- 唯一值数：`1247`
- 高频值：
  - `Conversation` 2254
  - `Stream of consciousness` 728
  - `Dialogue` 694
  - `Description of a place` 307
  - `Conversation/dialogue` 186

代表样本：

- `idx 0`: `Description of a character and their background`
- `idx 1`: `Stream of consciousness`
- `idx 11`: `Conversation/dialogue`
- `idx 98`: `Description of a place (room, chimney`

### 直接观察

存在以下不规则性：

- 斜杠合并类别：`Conversation/dialogue`
- 描述式长值：`Description of a character and their background`
- 括号未闭合：`Description of a place (room, chimney`

## 4.7 Active character

### 直接观察

- 出现数：`7009`
- 唯一值数：`6681`
- 高频值：
  - `The narrator` 52
  - `The protagonist` 29
  - `The narrator, the woman` 11
  - `The man, the woman` 9

代表样本：

- `idx 0`: `Matvey Savitch, Marfa Semyonovna Kapluntsev, Vasya`
- `idx 10`: `Narrator, Alessio (narrator's son), Pietro (narrator's husband), cousin`
- `idx 21`: `Padre Burgos, Padre Gomez, Padre Zamora, Francisco Zaldua`

### 直接观察

- 有时是专名列表。
- 有时是泛称：`The narrator`、`The protagonist`
- 有时混合说明性附注：`Alessio (narrator's son)`
- 少量样本括号未闭合：
  - `The narrator, Felipe (the guide`
  - `Alcalde (Mayor), Cura (Priest`

### 基于模式的推断

此字段可以先按逗号拆分为“角色字符串列表”，但不应直接视为可靠的人名标准化结果。

## 4.8 Fuzzy time

### 直接观察

- 出现数：`1534`
- 唯一值数：`668`
- 高频值：
  - `Nonspecific moment` 235
  - `Nonspecific` 107
  - `Nighttime` 73
  - `Evening` 58
  - `Morning` 27

代表样本：

- `idx 0`: `Ten years ago`
- `idx 4`: `Nonspecific moment`
- `idx 20`: `Midnight`
- `idx 22`: `Summer`

### 直接观察

存在以下边界情况：

- 模糊词：`Nonspecific`, `Unspecified`
- 字面空值：`None`
- 粗颗粒季节/时段：`Winter`, `Night`, `Dawn`

### 基于模式的推断

该字段更像“模糊时间提示”而非严格时间表达式，不适合直接标准化到 ISO 时间。

## 4.9 Fuzzy place

### 直接观察

- 出现数：`5513`
- 唯一值数：`4379`
- 高频值：
  - `Unnamed location` 159
  - `Unnamed place` 59
  - `Unnamed locations` 48
  - `Unnamed` 43
  - `Unnamed room` 35

代表样本：

- `idx 0`: `Little house, tallow and oil factory`
- `idx 2`: `Road, hostelries`
- `idx 10`: `Unnamed location in the countryside`
- `idx 19`: `Unnamed room`

### 直接观察

存在很多近义变体：

- `Unnamed location`
- `Unnamed place`
- `Unnamed locations`
- `Unnamed`

也存在多地点串联和未闭合括号：

- `Simla, Kumaul, Delhi`
- `Mutina (Mutina, an ancient city in Italy`
- `FLUTTERBY (a ship`

## 5. 缺失、变体、格式异常总结

### 直接观察

#### 缺失情况

- `Summary` 缺失：3 条
- `Tone` 缺失：12 条
- `Speech standard` 缺失：25 条
- `Literary form` 缺失：63 条
- `Enunciation` 缺失：78 条
- `Genre` 缺失：389 条
- `Active character` 缺失：1217 条
- `Fuzzy time` 缺失：6692 条
- `Fuzzy place` 缺失：2713 条

#### 值级异常

- 未闭合括号：
  - `Quoted character: Brenda, George (mentioned but not quoted`
  - `Absolute place: Holywood (church`
  - `Literary form: Description of a place (room, chimney`
- 字面空值：
  - `Quoted character: None`
  - `Quoted work: None`
  - `Absolute time: None`
- 斜杠混合类别：
  - `Mystery/Crime Fiction`
  - `Conversation/dialogue`
  - `Standard/neutral`

### 基于模式的推断

“可解析”不等于“完全规范”。该数据非常适合做字段抽取，但如果直接拿值做离散类别训练标签，噪声会比较大。

## 6. `analysis` 是否适合转成结构化 JSON

### 直接观察

适合。原因是：

- 每条都满足 `Label: value` 行级模式。
- 标签集合有限且稳定。
- 重点标签覆盖率很高。

### 基于模式的推断

转换为 JSON 时建议保留两层：

1. 原始字符串值
2. 规范化后的辅助字段

这样既保留原始文学描述，又能支持后续建模和统计。

## 7. 建议的规范化 JSON Schema

```json
{
  "instruction_id": "string",
  "full_text": "string",
  "analysis_raw": "string",
  "analysis": {
    "summary": "string|null",
    "enunciation_raw": "string|null",
    "enunciation_norm": "first_person|third_person|dialogue|mixed|other|null",
    "tone_raw": ["string"],
    "genre_raw": ["string"],
    "speech_standard_raw": ["string"],
    "literary_form_raw": ["string"],
    "active_characters_raw": ["string"],
    "fuzzy_time": "string|null",
    "fuzzy_place_raw": ["string"],
    "other_labels": {
      "Narrative arc": "string",
      "Literary movement": "string",
      "Time setting": "string",
      "Intertextuality": "string",
      "Trope": "string",
      "Absolute place": "string",
      "Diegetic time": "string",
      "Quoted character": "string",
      "Quoted work": "string",
      "Absolute time": "string"
    },
    "parsing_warnings": ["string"]
  }
}
```

## 8. Python 代码：加载、解析、统计、异常检测、规范化输出

```python
from datasets import load_from_disk
from collections import Counter
import re, json

DATASET_PATH = "."

TARGET_LABELS = [
    "Summary", "Enunciation", "Tone", "Genre", "Speech standard",
    "Literary form", "Active character", "Fuzzy time", "Fuzzy place"
]

LINE_RE = re.compile(r"^([^:]{1,80}):\\s*(.*)$")

def parse_analysis(text: str):
    parsed = {}
    warnings = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            warnings.append(f"unparsed_line::{line}")
            continue
        label, value = m.group(1).strip(), m.group(2).strip()
        if label in parsed:
            warnings.append(f"duplicate_label::{label}")
        parsed[label] = value
        if value.lower() in {"none", "n/a", "null"}:
            warnings.append(f"literal_null::{label}")
        if "(" in value and ")" not in value:
            warnings.append(f"unmatched_paren::{label}")
    return parsed, warnings

def split_listish(value: str):
    if not value or value.lower() in {"none", "n/a", "null"}:
        return []
    return [x.strip() for x in re.split(r",\\s*", value) if x.strip()]

def normalize_enunciation(v: str):
    if not v:
        return None
    s = v.lower()
    if s.startswith("third-person narr"):
        return "third_person"
    if s.startswith("first-person narr"):
        return "first_person"
    if "third-person" in s and "dialog" in s:
        return "mixed"
    if re.match(r"^(dialog|dialogue|conversation)", s) or "speaking" in s:
        return "dialogue"
    return "other"

def normalize_record(rec):
    parsed, warnings = parse_analysis(rec["analysis"])
    return {
        "instruction_id": rec["instruction_id"],
        "full_text": rec["full_text"],
        "analysis_raw": rec["analysis"],
        "analysis": {
            "summary": parsed.get("Summary"),
            "enunciation_raw": parsed.get("Enunciation"),
            "enunciation_norm": normalize_enunciation(parsed.get("Enunciation")),
            "tone_raw": split_listish(parsed.get("Tone", "")),
            "genre_raw": split_listish(parsed.get("Genre", "")),
            "speech_standard_raw": split_listish(parsed.get("Speech standard", "")),
            "literary_form_raw": split_listish(parsed.get("Literary form", "")),
            "active_characters_raw": split_listish(parsed.get("Active character", "")),
            "fuzzy_time": parsed.get("Fuzzy time"),
            "fuzzy_place_raw": split_listish(parsed.get("Fuzzy place", "")),
            "other_labels": {
                k: v for k, v in parsed.items() if k not in TARGET_LABELS
            },
            "parsing_warnings": warnings
        }
    }

def main():
    ds = load_from_disk(DATASET_PATH)
    n = len(ds)

    label_freq = Counter()
    value_freq = {k: Counter() for k in TARGET_LABELS}
    label_seq_freq = Counter()
    anomaly_examples = []

    for i, rec in enumerate(ds):
        parsed, warnings = parse_analysis(rec["analysis"])
        labels = list(parsed.keys())
        label_seq_freq[tuple(labels)] += 1

        for label in labels:
            label_freq[label] += 1
        for label in TARGET_LABELS:
            if label in parsed:
                value_freq[label][parsed[label]] += 1

        if warnings and len(anomaly_examples) < 20:
            anomaly_examples.append({
                "idx": i,
                "warnings": warnings,
                "analysis": rec["analysis"]
            })

    print("num_rows:", n)
    print("field_names:", ds.column_names)

    print("\\nTarget label frequency:")
    for label in TARGET_LABELS:
        c = label_freq[label]
        print(f"{label}\\t{c}\\t{c/n:.2%}")

    print("\\nTop all labels:")
    for k, v in label_freq.most_common():
        print(f"{k}\\t{v}")

    print("\\nTop exact label sequences:")
    for seq, c in label_seq_freq.most_common(10):
        print(c, " | ".join(seq))

    print("\\nRepresentative values for target labels:")
    for label in TARGET_LABELS:
        print(f"\\n## {label}")
        for val, c in value_freq[label].most_common(10):
            print(f"{c}\\t{val}")

    print("\\nAnomaly examples:")
    for ex in anomaly_examples[:10]:
        print(json.dumps(ex, ensure_ascii=False, indent=2))

    print("\\nNormalized example:")
    example = normalize_record(ds[0])
    print(json.dumps(example, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

## 9. 规范化输出示例

以 `idx 0` 为例，规范化后可表示为：

```json
{
  "instruction_id": "示例ID",
  "full_text": "示例文本",
  "analysis_raw": "Summary: The story is about an old widow and her sons, particularly one son named Vasya who breeds pigeons and sells them for extra money.\nEnunciation: Third-person narrative\nTone: Neutral\nGenre: Realistic fiction\nSpeech standard: Conversational\nLiterary form: Description of a character and their background\nActive character: Matvey Savitch, Marfa Semyonovna Kapluntsev, Vasya\nFuzzy time: Ten years ago\nFuzzy place: Little house, tallow and oil factory",
  "analysis": {
    "summary": "The story is about an old widow and her sons, particularly one son named Vasya who breeds pigeons and sells them for extra money.",
    "enunciation_raw": "Third-person narrative",
    "enunciation_norm": "third_person",
    "tone_raw": ["Neutral"],
    "genre_raw": ["Realistic fiction"],
    "speech_standard_raw": ["Conversational"],
    "literary_form_raw": ["Description of a character and their background"],
    "active_characters_raw": [
      "Matvey Savitch",
      "Marfa Semyonovna Kapluntsev",
      "Vasya"
    ],
    "fuzzy_time": "Ten years ago",
    "fuzzy_place_raw": [
      "Little house",
      "tallow and oil factory"
    ],
    "other_labels": {},
    "parsing_warnings": []
  }
}
```

## 10. 明确区分：直接观察 vs 基于模式推断

## 从数据直接观察得到的结论

- 数据集有 `8226` 条，三个字段都是字符串。
- `analysis` 在行级上全部可用 `Label: value` 解析。
- 标签集合稳定，观察到 `19` 个标签。
- 重点标签覆盖率不一致，`Fuzzy time` 明显稀疏。
- 标签顺序不是完全一致，共有 `1391` 种不同标签序列。
- 值中存在缺失、`None`、未闭合括号、斜杠复合类别等现象。

## 基于模式做出的推断

- 数据集适合转换为结构化 JSON。
- `Enunciation` 可以安全地粗归并为少数几类，但应保留原值。
- `Tone/Genre/Speech standard/Literary form` 更适合作为“原值列表”而不是严格闭集。
- `Active character` 可以先切成字符串列表，但不应直接作为标准化实体识别结果。
- `Fuzzy time` 和 `Fuzzy place` 更像“文学模糊标注”，不应强行映射成严格时空知识图谱字段。

## 11. 用于 LLM 微调或文学标注研究的预处理建议

### 建议 1：保留原始 `analysis`

不要只保留规范化 JSON，建议同时保留：

- `analysis_raw`
- `analysis_json`

这样便于回溯和误差分析。

### 建议 2：先做弱规范化

建议先做：

- 标签名规范化
- `None/N/A/null` 转 `null`
- 逗号拆分列表字段
- `Enunciation` 粗分类

不建议一开始就做过强语义归并。

### 建议 3：对异常样本单独打标

建议额外记录：

- `missing_required_fields`
- `literal_null_fields`
- `unmatched_parenthesis_fields`
- `freeform_enunciation`

### 建议 4：训练任务分层

可拆成两类任务：

1. 原始风格生成  
   输入：`full_text`  
   输出：原始 `analysis`

2. 结构化抽取  
   输入：`full_text`  
   输出：规范化 JSON

### 建议 5：研究场景下先限定核心字段

若做文学标注研究，建议优先使用：

- `Summary`
- `Enunciation`
- `Tone`
- `Genre`
- `Literary form`
- `Active character`

将 `Fuzzy time` 和 `Fuzzy place` 作为可选补充字段。

### 建议 6：评估时分开报告

建议至少报告以下指标：

- 行级解析成功率
- 字段级缺失率
- 值级规范化覆盖率
- 归一化后类别分布
- 异常样本占比

## 12. 结论

### 直接观察

`Pclanglais/Brahe-Novels` 的 `analysis` 字段是一个高可解析度的半结构化文学标注字段。其行级格式非常稳定，适合程序化解析；但值级表达并不完全标准化，存在同义变体、并列值、说明性补充、字面空值和少量格式残缺。

### 基于模式的推断

这个数据集非常适合：

- 结构化信息抽取
- 文学标签生成
- LLM 从自由文本到半结构化标注的监督微调
- 文学元数据规范化研究

但如果要把它当作严格分类标签数据集，必须先进行系统的值归一化与异常样本标记。
