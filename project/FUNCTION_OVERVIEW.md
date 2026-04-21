# 项目函数总览

## 1. 项目概览

这个项目是一条面向 Brahe-Novels 数据的文本分类准备与基线实验流水线。核心流程是：

1. 从 Hugging Face `datasets` 格式的本地数据集中读取原始记录。
2. 解析每条记录里的 `analysis` 多行文本，把 `Summary: ...`、`Genre: ...`、`Enunciation: ...` 等标签拆成结构化字段。
3. 将原始 `enunciation_raw` 和 `genre_raw` 标签归一化为少量可训练的分类标签。
4. 导出分类用 CSV。
5. 用多数类基线和 TF-IDF + Logistic Regression 训练/评估分类器，并保存指标、预测、错误样本和混淆矩阵。

项目没有复杂类层次，主要由 4 个 `src/` Python 脚本组成。`tests/` 目录中的单元测试覆盖解析、标签归一化和基线训练的关键行为。

## 2. 目录/模块说明

- `src/parse_brahe_analysis.py`：把原始 `analysis` 字符串解析成结构化 JSONL/CSV 记录，是数据准备的第一步。
- `src/normalize_enunciation.py`：把 `enunciation_raw` 归一化到 `first_person`、`third_person`、`dialogue`、`mixed`、`epistolary`、`other`。
- `src/normalize_genre.py`：把 `genre_raw` 归一化到较少的顶层体裁标签，并标记是否适合进入分类训练。
- `src/baseline_classification.py`：读取分类 CSV，做分层 train/dev/test 切分，训练多数类和 TF-IDF 逻辑回归基线，输出评估产物。
- `tests/`：单元测试，描述了每个模块的预期行为和边界情况。
- `data/`：本地数据和派生结果目录，不属于源码分析重点。
- `outputs/`：模型评估输出目录，不属于源码分析重点。

## 3. 执行流程总结

常见端到端执行顺序推测如下，这是基于脚本默认参数和文件名的代码行为推断：

```bash
python -m src.parse_brahe_analysis \
  --dataset-path data/brahe_novels \
  --output data/brahe_novels_parsed.jsonl

python -m src.normalize_enunciation \
  --input data/brahe_novels_parsed.jsonl \
  --output data/brahe_novels_enunciation_normalized.jsonl \
  --classification-output data/brahe_enunciation_classification.csv

python -m src.normalize_genre \
  --input data/brahe_novels_parsed.jsonl \
  --output data/brahe_novels_genre_normalized.jsonl \
  --classification-output data/brahe_genre_classification.csv

python -m src.baseline_classification \
  --output-root outputs/baselines
```

整体调用关系：

`load_rows()` -> `parse_record()` -> `parse_analysis()` -> parsed JSONL/CSV

parsed JSONL -> `normalize_records()` -> `normalize_enunciation()` 或 `normalize_genre()` -> normalized JSONL + classification CSV

classification CSV -> `load_classification_csv()` -> `make_stratified_splits()` -> `train_majority_baseline()` / `train_tfidf_logreg()` -> `evaluate_predictions()` + `save_confusion_matrix()`

## 4. 各文件说明

## `src/parse_brahe_analysis.py`

**文件作用：**

该文件负责把 Brahe-Novels 数据集中每条记录的 `analysis` 字段从多行 `Label: value` 文本解析成结构化字段。它既可以导出 JSONL，也可以导出 CSV，是后续标签归一化模块的数据来源。

**重要常量：**

- `LABEL_RE`：匹配形如 `Label: value` 的行，标签最长 80 个非冒号/非换行字符。
- `KNOWN_LABELS`：把已知标签映射到输出字段名，例如 `Genre` -> `genre_raw`，`Speech standard` -> `speech_standard_raw`。
- `OUTPUT_KEYS`：所有标准输出字段。缺失标签会保留为 `None`，方便后续处理有稳定 schema。

### `snake_case(label: str) -> str`

- 作用：把未知标签转成 snake_case 字段名。
- 输入：标签字符串，例如 `Speech standard` 或 `Narrative arc`。
- 输出：小写、以下划线分隔的字段名，例如 `speech_standard`、`narrative_arc`。
- 关键逻辑：去除首尾空白，转小写，把非字母数字字符替换成 `_`，再去掉两端 `_`。
- 与其他部分的关系：`parse_analysis()` 遇到不在 `KNOWN_LABELS` 中的标签时调用它，并把结果放进 `other_labels`。

### `parse_analysis(analysis: str | None) -> dict[str, Any]`

- 作用：解析原始多行分析文本，返回包含标准字段和 `other_labels` 的字典。
- 输入：可能为 `None` 的 `analysis` 字符串。典型格式是多行 `Summary: ...`、`Genre: ...`。
- 输出：字典，包含 `summary`、`enunciation_raw`、`tone_raw`、`genre_raw` 等字段；未出现的标准字段为 `None`；未知标签进入 `other_labels`。
- 关键逻辑：
  - 初始化所有标准字段为 `None`。
  - 逐行匹配 `LABEL_RE`。
  - 已知标签直接写入对应标准字段。
  - 未知标签转 snake_case 后写入 `other_labels`。
  - 不含新标签的连续行会追加到上一个标签值后面，保留换行。
- 与其他部分的关系：`parse_record()` 直接调用它；测试中确认了缺失字段、未知标签和多行值的行为。

### `parse_record(row: dict[str, Any]) -> dict[str, Any]`

- 作用：把数据集中的一条原始 row 转成后续 JSONL/CSV 使用的完整记录。
- 输入：包含 `instruction_id`、`full_text`、`analysis` 的字典。
- 输出：保留源字段并合并 `parse_analysis()` 的解析结果。
- 关键逻辑：先构造源字段子集，再 `record.update(parse_analysis(...))`。
- 与其他部分的关系：`convert_dataset()` 和 `main()` 都通过它生成输出记录。

### `load_rows(dataset_path: str | Path) -> Iterable[dict[str, Any]]`

- 作用：从本地 Hugging Face dataset 目录读取记录。
- 输入：数据集路径，默认是 `data/brahe_novels`。
- 输出：逐条 yield 普通 Python `dict`。
- 关键逻辑：
  - 动态导入 `datasets.load_from_disk`。
  - 如果加载结果有 `train` split，则读取 `dataset["train"]`；否则直接遍历 dataset。
  - 缺少 `datasets` 包时用 `SystemExit` 给出安装提示。
- 与其他部分的关系：`convert_dataset()` 和 `main()` 的数据入口。

### `write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> int`

- 作用：把记录写成 JSON Lines。
- 输入：记录迭代器、输出路径。
- 输出：写入的记录数。
- 关键逻辑：每条记录用 `json.dumps(..., ensure_ascii=False)` 写一行。
- 与其他部分的关系：供 `convert_dataset()` 和 `main()` 根据输出后缀选择。

### `write_csv(records: Iterable[dict[str, Any]], output_path: str | Path) -> int`

- 作用：把解析记录写成 CSV。
- 输入：记录迭代器、输出路径。
- 输出：写入的记录数。
- 关键逻辑：固定 CSV 字段顺序；`other_labels` 作为 JSON 字符串写入单个单元格。
- 与其他部分的关系：供 `convert_dataset()` 和 `main()` 在输出路径以 `.csv` 结尾时使用。

### `convert_dataset(dataset_path: str | Path, output_path: str | Path) -> int`

- 作用：封装“加载数据集 -> 解析记录 -> 写文件”的转换过程。
- 输入：数据集目录和输出文件路径。
- 输出：写入记录数。
- 关键逻辑：根据 `output_path` 后缀选择 `write_csv()` 或 `write_jsonl()`。
- 与其他部分的关系：适合被其他脚本/测试复用；命令行 `main()` 实现了类似流程并额外打印示例。

### `main() -> None`

- 作用：命令行入口。
- 输入：命令行参数 `--dataset-path`、`--output`、`--examples`。
- 输出：写入文件，并在终端打印写入数量和前几个解析示例。
- 关键逻辑：先把所有 row 读入列表，再根据输出后缀选择 writer。
- 与其他部分的关系：当脚本作为 `python -m src.parse_brahe_analysis` 或直接运行时执行。

### `_append_line(existing: Any, line: str) -> str`

- 作用：辅助函数，把非标签行追加到当前标签的值中。
- 输入：已有值和新行。
- 输出：如果已有值为空，返回新行；否则返回 `existing + "\n" + line`。
- 关键逻辑：保留原始多行结构。
- 与其他部分的关系：只被 `parse_analysis()` 使用。

## `src/normalize_enunciation.py`

**文件作用：**

该文件把解析后的 `enunciation_raw` 文本标签归一化为较稳定的叙述方式分类标签，并把高置信度样本导出成二列分类 CSV。它同时输出频率表和不确定样本，方便人工复查规则。

**重要常量：**

- `LABELS`：允许的归一化标签集合。
- `DIRECT_ALIASES`：精确匹配的别名字典，例如 `third person narrative` -> `third_person`。
- `EPISTOLARY_TERMS`、`DIALOGUE_TERMS`、`WEAK_DIALOGUE_TERMS`、`FIRST_PERSON_TERMS`、`THIRD_PERSON_TERMS`、`MIXED_TERMS`、`NARRATIVE_TERMS`：规则匹配使用的关键词组。

### `normalize_enunciation(raw: str | None) -> dict[str, Any]`

- 作用：把一个原始 enunciation 值映射为归一化标签，并标记是否不确定。
- 输入：`enunciation_raw`，可以为 `None` 或空字符串。
- 输出：包含 `enunciation_raw`、`enunciation_norm`、`enunciation_uncertain`、`enunciation_norm_reason` 的字典。
- 关键逻辑：
  - 缺失值归为 `other`，并标记不确定。
  - 先清洗文本，再查 `DIRECT_ALIASES`。
  - 之后按规则检测 epistolary、mixed、dialogue、weak dialogue、first person、third person。
  - 优先级很重要：例如包含 `letter` 会优先归为 `epistolary`；同时包含叙述和对话会归为 `mixed`。
- 与其他部分的关系：`normalize_records()` 对每条记录调用它；`write_classification_csv()` 只保留非 `other` 且不确定标记为 False 的样本。

### `clean_value(value: str) -> str`

- 作用：统一原始标签文本格式，便于规则匹配。
- 输入：任意字符串。
- 输出：小写、去标点、合并空白后的字符串。
- 关键逻辑：把连字符替换为空格，只保留字母数字，多个空白合并为一个。
- 与其他部分的关系：`normalize_enunciation()` 的第一步。

### `read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]`

- 作用：读取解析后的 JSONL 文件。
- 输入：JSONL 路径。
- 输出：逐条 yield 字典记录。
- 关键逻辑：跳过空行，用 `json.loads()` 解析非空行。
- 与其他部分的关系：`main()` 的输入读取函数。

### `write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int`

- 作用：写出归一化后的完整记录。
- 输入：记录迭代器、输出路径。
- 输出：写入条数。
- 与其他部分的关系：`main()` 用它生成 `data/brahe_novels_enunciation_normalized.jsonl`。

### `write_classification_csv(records: Iterable[dict[str, Any]], path: str | Path) -> int`

- 作用：生成机器学习分类器使用的 `text,label` CSV。
- 输入：已归一化记录。
- 输出：写入的干净样本数。
- 关键逻辑：调用 `is_clean_classification_record()` 过滤掉 `other` 和不确定样本；`text` 来自 `full_text`，`label` 来自 `enunciation_norm`。
- 与其他部分的关系：输出文件是 `baseline_classification.py` 的默认输入之一。

### `write_frequencies(counter: Counter[str], path: str | Path) -> None`

- 作用：写出原始 enunciation 值的频率表。
- 输入：`Counter`，键是原始值，值是出现次数。
- 输出：CSV 文件，列为 `enunciation_raw,count`。
- 与其他部分的关系：帮助人工检查规则覆盖率。

### `write_uncertain(records: Iterable[dict[str, Any]], path: str | Path) -> int`

- 作用：导出不确定的 enunciation 归一化案例。
- 输入：已归一化记录。
- 输出：写出的不确定样本数。
- 关键逻辑：只写 `record["enunciation_uncertain"]` 为 True 的样本，保留 `instruction_id`、原始值、归一化标签和原因。
- 与其他部分的关系：用于人工复核，可能进一步改进规则。

### `normalize_records(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]`

- 作用：批量归一化记录。
- 输入：解析后的记录迭代器。
- 输出：逐条 yield 合并了 enunciation 归一化字段的新字典。
- 关键逻辑：复制原记录，避免原地修改传入对象。
- 与其他部分的关系：`main()` 用它创建完整归一化数据。

### `is_clean_classification_record(record: dict[str, Any]) -> bool`

- 作用：判断一条记录是否适合进入 enunciation 分类训练。
- 输入：归一化记录。
- 输出：布尔值。
- 关键逻辑：`enunciation_norm` 不能是 `other`，且 `enunciation_uncertain` 必须为 False。
- 与其他部分的关系：被 `write_classification_csv()` 调用；测试明确验证了过滤规则。

### `_contains_any(text: str, terms: tuple[str, ...]) -> bool`

- 作用：辅助函数，判断文本中是否包含任一关键词。
- 输入：清洗后的文本和关键词元组。
- 输出：布尔值。
- 与其他部分的关系：只被 `normalize_enunciation()` 使用。

### `_result(raw: str | None, label: str, uncertain: bool, reason: str) -> dict[str, Any]`

- 作用：统一构造 enunciation 归一化结果。
- 输入：原始值、标签、不确定标记、规则原因。
- 输出：标准结果字典。
- 关键逻辑：先检查标签是否在 `LABELS` 中，不合法则抛 `ValueError`。
- 与其他部分的关系：所有 `normalize_enunciation()` 分支都通过它返回。

### `main() -> None`

- 作用：命令行入口，完成 enunciation 归一化和多个输出文件生成。
- 输入：`--input`、`--output`、`--classification-output`、`--frequency-output`、`--uncertain-output`、`--examples`。
- 输出：归一化 JSONL、分类 CSV、频率 CSV、不确定样本 CSV，并打印统计和示例。
- 与其他部分的关系：连接 `read_jsonl()`、`normalize_records()`、各类 writer。

## `src/normalize_genre.py`

**文件作用：**

该文件把 `genre_raw` 归一化为项目定义的顶层体裁标签。与 enunciation 不同，它额外处理复合体裁：如果多个候选都映射到同一个顶层标签，就保留；如果映射到多个顶层标签，则标记为不确定并从分类 CSV 中剔除。

**重要常量：**

- `LABELS`：允许的体裁标签集合，包括 `historical`、`drama`、`romance`、`speculative`、`other` 等。
- `LABEL_PRIORITY`：理论上用于在多个标签中选择主标签的优先顺序。
- `DIRECT_ALIASES`：大量精确别名映射，例如 `detective fiction` -> `mystery_crime`。
- `SPLIT_RE`：用于把复合 genre 字符串按 `/`、`,`、`;`、`+`、`and`、`&` 拆成候选。

### `normalize_genre(raw: str | None) -> dict[str, Any]`

- 作用：把一个原始 genre 值映射成归一化体裁标签，并决定是否保留进分类数据。
- 输入：`genre_raw`，可以为空。
- 输出：包含 `genre_raw`、`genre_candidates`、`genre_candidate_labels`、`genre_norm`、`genre_keep`、`genre_uncertain`、`genre_norm_reason` 的字典。
- 关键逻辑：
  - 缺失值归为 `other`，`genre_keep=False`，`genre_uncertain=True`。
  - 先用 `parse_genre_candidates()` 拆分候选。
  - 每个候选经 `_map_candidate()` 映射到顶层标签或 `None`。
  - 没有任何候选匹配规则时归为 `other` 并标记不确定。
  - 所有候选都指向同一个顶层标签时保留。
  - 多个不同顶层标签时选第一个匹配标签为 `genre_norm`，但 `genre_keep=False` 且 `genre_uncertain=True`。
- 与其他部分的关系：`normalize_records()` 批量调用；`write_classification_csv()` 只导出 `genre_keep=True` 的确定样本。

### `parse_genre_candidates(raw: str) -> list[str]`

- 作用：把复合 genre 文本拆成候选短语。
- 输入：原始 genre 字符串。
- 输出：清洗后的候选列表。
- 关键逻辑：先 `clean_value()`，再用 `SPLIT_RE` 按多种分隔符拆分；如果拆不出 parts 但清洗后非空，则返回单元素列表。
- 与其他部分的关系：`normalize_genre()` 依赖它识别复合标签；测试覆盖了逗号和斜杠拆分。

### `clean_value(value: str) -> str`

- 作用：清洗 genre 文本，保留可能作为分隔符的 `/&,+;`。
- 输入：任意字符串。
- 输出：小写、去除无关标点、合并空白后的字符串。
- 关键逻辑：连字符转空格；非字母数字和非分隔符字符替换为空格。
- 与其他部分的关系：`parse_genre_candidates()` 使用它。

### `read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]`

- 作用：读取解析后的 JSONL。
- 输入：文件路径。
- 输出：逐条 yield 字典。
- 与其他部分的关系：`main()` 的输入读取函数。

### `normalize_records(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]`

- 作用：批量归一化 genre。
- 输入：解析记录。
- 输出：合并 genre 归一化字段的新记录。
- 关键逻辑：复制原记录再更新，避免原地修改。
- 与其他部分的关系：`main()` 的核心转换步骤。

### `write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int`

- 作用：写出 genre 归一化后的完整记录。
- 输入：记录迭代器和输出路径。
- 输出：写入条数。

### `write_classification_csv(records: Iterable[dict[str, Any]], path: str | Path) -> int`

- 作用：导出 genre 分类训练 CSV。
- 输入：已归一化记录。
- 输出：写入的干净样本数。
- 关键逻辑：只保留 `is_clean_classification_record()` 返回 True 的记录；输出列为 `text,label`。
- 与其他部分的关系：输出文件是 `baseline_classification.py` 的默认输入之一。

### `write_frequencies(counter: Counter[str], path: str | Path) -> None`

- 作用：导出原始 genre 值频率表。
- 输入：原始值计数器。
- 输出：`genre_raw,count` CSV。
- 与其他部分的关系：帮助人工判断哪些原始值需要新增规则。

### `write_uncertain(records: Iterable[dict[str, Any]], path: str | Path) -> int`

- 作用：导出不确定或被丢弃的 genre 样本。
- 输入：已归一化记录。
- 输出：写出的案例数。
- 关键逻辑：当 `genre_uncertain` 为 True 或 `genre_keep` 为 False 时写出，包含候选列表、候选标签、最终标签和原因。
- 与其他部分的关系：人工复核和规则迭代入口。

### `is_clean_classification_record(record: dict[str, Any]) -> bool`

- 作用：判断记录是否适合进入 genre 分类训练。
- 输入：归一化记录。
- 输出：布尔值。
- 关键逻辑：要求 `genre_keep is True`、`genre_norm != "other"`、`genre_uncertain` 为 False，并且 `full_text` 非空。
- 与其他部分的关系：`write_classification_csv()` 使用；测试覆盖了保留和剔除案例。

### `_map_candidate(candidate: str) -> str | None`

- 作用：把单个 genre 候选短语映射到顶层标签。
- 输入：已经清洗的小写候选，例如 `historical fiction`、`science fiction`。
- 输出：顶层标签字符串，或无法匹配时返回 `None`。
- 关键逻辑：
  - 先查 `DIRECT_ALIASES`。
  - 再用关键词规则匹配 historical、drama、romance、adventure、poetry、children/YA、mystery/crime、comedy/satire、speculative、nonfiction/essay、religious/philosophical。
- 与其他部分的关系：`normalize_genre()` 对每个候选调用它。

### `_choose_primary_label(candidates: list[str], mapped: list[str | None]) -> str`

- 作用：在复合 genre 的多个映射结果中选择一个主标签。
- 输入：候选列表和对应映射结果。
- 输出：第一个非空映射标签；如果都为空则返回 `other`。
- 关键逻辑：当前实现会先返回 `mapped` 中第一个非 `None` 标签，因此后面的 `LABEL_PRIORITY` 分支实际上不会在有非空标签时执行。这一点是代码行为，不是注释推断。
- 与其他部分的关系：`normalize_genre()` 在多标签冲突时用它填充 `genre_norm`，但同时把记录标记为不确定和不保留。

### `_unique(values: Iterable[str]) -> list[str]`

- 作用：按出现顺序去重。
- 输入：标签序列。
- 输出：不重复标签列表。
- 关键逻辑：用 `seen` 集合记录已出现值，同时保留原始顺序。
- 与其他部分的关系：`normalize_genre()` 用它判断是否存在多个不同顶层标签。

### `_result(...) -> dict[str, Any]`

- 作用：统一构造 genre 归一化结果。
- 输入：原始值、候选列表、候选标签、最终标签、保留标记、不确定标记、原因。
- 输出：标准结果字典。
- 关键逻辑：验证最终标签必须在 `LABELS` 中，否则抛 `ValueError`。
- 与其他部分的关系：所有 `normalize_genre()` 分支都通过它返回。

### `main() -> None`

- 作用：命令行入口，完成 genre 归一化和输出。
- 输入：`--input`、`--output`、`--classification-output`、`--frequency-output`、`--uncertain-output`、`--examples`。
- 输出：归一化 JSONL、分类 CSV、频率 CSV、不确定/丢弃样本 CSV，并打印统计和示例。
- 与其他部分的关系：连接读取、归一化、写文件和统计打印。

## `src/baseline_classification.py`

**文件作用：**

该文件用于复现实验基线：读取 `text,label` 格式分类 CSV，做分层切分，训练两个模型，并输出完整评估结果。它不负责标签清洗，而是假设输入 CSV 已由 normalize 脚本生成。

**重要常量/类：**

- `DEFAULT_DATASETS`：默认运行两个任务：`enunciation` 和 `genre`。
- `MPLCONFIGDIR` 和 `matplotlib.use("Agg")`：使 matplotlib 在无图形界面的环境中也能保存图片。

### `DatasetSpec`

- 类职责：保存一个分类数据集的名字和路径。
- 重要属性：
  - `name: str`：数据集名，用作输出子目录名。
  - `path: Path`：CSV 路径。
- 关键特性：`@dataclass(frozen=True)`，实例创建后不可修改。
- 与其他部分的关系：`parse_dataset_specs()` 创建它，`run_dataset()` 消费它。

### `load_classification_csv(path: str | Path) -> pd.DataFrame`

- 作用：读取并清理分类 CSV。
- 输入：包含 `text` 和 `label` 列的 CSV 路径。
- 输出：只含 `text,label` 两列的干净 `DataFrame`。
- 关键逻辑：
  - 如果缺少 `text` 或 `label` 列，抛 `ValueError`。
  - 删除缺失值。
  - 强制转字符串。
  - 删除空白文本或空白标签。
  - 重置索引。
- 与其他部分的关系：`run_dataset()` 的第一步。

### `make_stratified_splits(...) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

- 作用：把数据分成 train/dev/test，并保持各标签比例。
- 输入：完整 DataFrame、随机种子、train/dev/test 比例，默认 70/15/15。
- 输出：三个重置索引后的 DataFrame。
- 关键逻辑：
  - 检查三个比例和必须为 1.0。
  - 每个标签至少需要 4 条，否则分层切分可能失败，所以直接抛 `ValueError`。
  - 第一次 `train_test_split` 切出 train 和临时集。
  - 第二次在临时集中按相对比例切出 dev/test。
- 与其他部分的关系：`run_dataset()` 用它产生并保存 splits；测试验证了可复现性、行数和标签覆盖。

### `train_majority_baseline(train_df: pd.DataFrame) -> DummyClassifier`

- 作用：训练多数类基线。
- 输入：训练集 DataFrame。
- 输出：已 fit 的 `DummyClassifier`。
- 关键逻辑：`strategy="most_frequent"`，预测永远是训练集中最多的标签。
- 与其他部分的关系：`run_dataset()` 会把它和 TF-IDF 模型一起评估；测试验证预测多数类。

### `train_tfidf_logreg(train_df: pd.DataFrame, seed: int) -> Pipeline`

- 作用：训练一个简单但可复现的文本分类模型。
- 输入：训练集 DataFrame 和随机种子。
- 输出：已 fit 的 scikit-learn `Pipeline`。
- 关键逻辑：
  - `TfidfVectorizer`：小写、Unicode accent stripping、1-gram 和 2-gram、`min_df=2`、最多 50000 特征。
  - `LogisticRegression`：`max_iter=1000`、`class_weight="balanced"`、固定随机种子。
- 与其他部分的关系：`run_dataset()` 保存这个模型并在 dev/test 上评估。

### `evaluate_predictions(split_df, predictions, labels) -> tuple[...]`

- 作用：计算预测指标、分类报告和错误样本表。
- 输入：评估 split、预测标签迭代器、完整标签列表。
- 输出：
  - `metrics`：包含 `accuracy` 和 `macro_f1`。
  - `report`：`classification_report(..., output_dict=True)` 的结果。
  - `misclassified`：包含 `text,label,prediction` 的错误样本 DataFrame。
- 关键逻辑：
  - `zero_division=0` 避免某些标签没有预测时产生异常。
  - 用 gold/pred 不相等筛出错误样本，并记录模型预测。
- 与其他部分的关系：`run_dataset()` 每个模型、每个 split 都调用；测试验证会返回错误行。

### `save_confusion_matrix(split_df, predictions, labels, output_path, title) -> None`

- 作用：保存混淆矩阵 PNG。
- 输入：评估 split、预测、标签顺序、输出路径、图标题。
- 输出：图片文件。
- 关键逻辑：
  - 用固定标签顺序计算 confusion matrix。
  - 根据标签数调整图宽，最小 7，最大 16。
  - 用 `ConfusionMatrixDisplay` 绘图并保存为 160 dpi PNG。
- 与其他部分的关系：`run_dataset()` 在 dev/test 评估时调用。

### `run_dataset(spec: DatasetSpec, output_root: str | Path, seed: int) -> pd.DataFrame`

- 作用：对单个数据集完成完整基线实验。
- 输入：数据集配置、输出根目录、随机种子。
- 输出：包含所有模型/split 指标的 summary DataFrame。
- 关键逻辑：
  - 创建数据集输出目录和 `splits/`。
  - 加载 CSV，分层切分并保存 `train.csv`、`dev.csv`、`test.csv`。
  - 保存标签列表 `labels.json`。
  - 训练 `majority` 和 `tfidf_logreg` 两个模型。
  - 对 dev/test 预测，保存 metrics JSON、classification report JSON/CSV、predictions CSV、misclassified CSV、confusion matrix PNG。
  - 保存 `summary_metrics.csv`。
- 与其他部分的关系：`main()` 对每个 `DatasetSpec` 调用它。

### `parse_dataset_specs(values: list[str] | None) -> list[DatasetSpec]`

- 作用：解析命令行传入的数据集规格。
- 输入：`--dataset` 参数列表，每项格式应为 `NAME=PATH`；也可以为 `None`。
- 输出：`DatasetSpec` 列表。
- 关键逻辑：
  - 未传参数时返回 `DEFAULT_DATASETS`。
  - 传入值不含 `=` 时抛 `ValueError`。
  - `name` 和 `path` 都会 strip 空白。
- 与其他部分的关系：`main()` 通过它决定要跑哪些数据集。

### `_write_json(path: str | Path, data: object) -> None`

- 作用：统一写 JSON 文件。
- 输入：路径和任意可 JSON 序列化数据。
- 输出：写入文件。
- 关键逻辑：`ensure_ascii=False`，`indent=2`。
- 与其他部分的关系：`run_dataset()` 保存指标和报告时使用。

### `main() -> None`

- 作用：命令行入口，运行一个或多个数据集的基线实验。
- 输入：`--dataset`、`--output-root`、`--seed`。
- 输出：各数据集独立输出目录，以及合并的 `all_summary_metrics.csv`。
- 关键逻辑：遍历 `DatasetSpec`，调用 `run_dataset()`，打印每个 summary，最后 concat 所有 summary。
- 与其他部分的关系：当脚本直接执行时作为总入口。

## `tests/test_parse_analysis.py`

**文件作用：**

测试 `parse_brahe_analysis.py` 的解析行为。它不是生产代码，但能帮助理解解析器承诺的行为。

**主要测试类：**

### `ParseAnalysisTests`

- 类职责：验证 `parse_analysis()` 和 `parse_record()`。
- 关键测试：
  - `test_parses_known_labels_to_snake_case_raw_keys`：确认所有已知标签映射到预期字段。
  - `test_missing_labels_are_present_as_none`：确认缺失标准字段保留为 `None`。
  - `test_multiline_values_attach_to_previous_label`：确认非标签行追加到前一个标签。
  - `test_unrecognized_labels_are_kept_in_other_labels`：确认未知标签进入 `other_labels`。
  - `test_parse_record_preserves_source_fields_and_adds_parsed_fields`：确认源字段和解析字段合并。

## `tests/test_normalize_enunciation.py`

**文件作用：**

测试 enunciation 归一化规则和训练样本过滤规则。

**主要测试类：**

### `NormalizeEnunciationTests`

- 类职责：验证 `normalize_enunciation()` 和 `is_clean_classification_record()`。
- 关键测试：
  - first-person、third-person、dialogue、mixed、epistolary 的典型映射。
  - 弱对话词如 `speaking` 会归为 `dialogue` 但标记不确定。
  - 缺失或无法识别值归为 `other` 且不确定。
  - 分类记录过滤会排除 `other` 和不确定样本。

## `tests/test_normalize_genre.py`

**文件作用：**

测试 genre 候选拆分、单标签归一化、复合标签处理和分类样本过滤。

**主要测试类：**

### `NormalizeGenreTests`

- 类职责：验证 `normalize_genre()`、`parse_genre_candidates()`、`is_clean_classification_record()`。
- 关键测试：
  - `Mystery/Crime Fiction, Thriller` 会拆成三个候选。
  - 常见单标签映射到 historical、drama、speculative、mystery_crime。
  - 多个候选映射到同一标签时保留。
  - 多个候选映射到不同标签时标记不确定并丢弃。
  - 罕见或缺失 genre 归为 `other` 并丢弃。

## `tests/test_baseline_classification.py`

**文件作用：**

测试基线分类模块的数据切分、简单模型和评估结果。

**主要测试类：**

### `BaselineClassificationTests`

- 类职责：验证 `make_stratified_splits()`、`train_majority_baseline()`、`evaluate_predictions()`。
- 关键测试：
  - 分层切分在固定 seed 下可复现，60 条样本按默认比例得到 42/9/9，并且每个 split 覆盖所有标签。
  - 多数类基线总是预测训练集多数标签。
  - 评估函数返回 accuracy、macro_f1、classification report，并正确列出误分类样本。

## 5. 基于推断或需要人工确认的部分

- 端到端命令顺序是根据默认输入/输出文件名和模块职责推断的，代码中没有单独的总控 pipeline 脚本。
- `genre` 复合标签冲突时，`_choose_primary_label()` 当前实际选择第一个匹配标签，而不是按 `LABEL_PRIORITY` 选择；如果作者原意是优先级选择，这里可能需要人工确认。
- `parse_brahe_analysis.py` 的 `convert_dataset()` 是可复用封装，但命令行 `main()` 没直接调用它，而是重复实现了类似逻辑；这不影响行为，但可能是后续可整理点。
- `data/result.md`、`data/` 中的数据文件和 `outputs/` 中的实验产物没有作为源码逐函数分析；它们更像输入/输出材料。
