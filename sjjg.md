# CostMonitoring 数据结构梳理

本文基于当前项目代码实现，对系统内已经稳定存在并被算法、报表、标注、Skills、拆分件逻辑实际消费的数据结构做统一梳理，便于后续部署到腾讯云服务器并接入 MySQL 或 PostgreSQL。

> 说明
>
> 1. 当前 `process_dataframe()` 会保留源文件中的额外列，因此运行时 DataFrame 可能比本文列出的字段更多。
> 2. 本文列出的字段是“系统已定义且稳定使用”的核心结构，适合作为数据库正式建模依据。
> 3. 表中的“主键/唯一/索引建议”是面向正式数据库部署的推荐方案，不代表当前 Pandas 已显式建立约束。

## 1. 核心成本数据表

建议在数据库中将原始标准化后的成本数据落为 `core_cost_records` 表。

### 1.1 表级建议

| 项目 | 建议 |
|---|---|
| 表名 | `core_cost_records` |
| 主键 | `cost_record_id`（自增或 UUID） |
| 唯一键 | `物料编码 + 工厂 + monitor_date + 成本金额` |
| 常用索引 | `物料编码`、`备件简称`、`适用车系`、`一级总成料号`、`物料编码 + 工厂 + monitor_date` |

### 1.2 字段定义

| 字段 | 当前 Pandas 类型 | 建议 SQL 类型 | 必填 | 说明 | 主键/唯一/索引建议 |
|---|---|---|---|---|---|
| 物料编码 | `object(str)` | `VARCHAR(64)` | 是 | 核心业务编码，所有报表和异常检测都依赖此字段 | 唯一键组成列，索引 |
| 成本金额 | `float64` | `DECIMAL(18,4)` | 是 | 当前代码中的动态价格列，可能来自 `价格` / `成本` / `单价` / `Price` / `Cost` / `含税价` / `未税价` | 唯一键组成列 |
| monitor_date | `datetime64[ns]` | `DATE` 或 `TIMESTAMP` | 是 | 标准化后的成本生效日期，系统所有时序逻辑都基于该字段 | 唯一键组成列，索引 |
| 工厂 | `object(str)` | `VARCHAR(64)` | 是 | 缺失时默认填充为 `总装` | 唯一键组成列，索引 |
| 物料名称 | `object(str)` | `VARCHAR(255)` | 否 | 缺失时默认填充为 `未知` | 可选索引 |
| 适用车系 | `object(str)` | `VARCHAR(128)` | 否 | 缺失时默认填充为 `未知` | 索引 |
| 备件简称 | `object(str)` | `VARCHAR(128)` | 否 | 缺失时默认填充为 `未知`；异常检测按该字段分组 | 索引 |
| 一级总成料号 | `object(str)` | `VARCHAR(64)` | 否 | 拆分件关系中的父级连接字段 | 索引 |
| 一级总成品名描述 | `object(str)` | `VARCHAR(255)` | 否 | 一级总成描述信息 | 否 |
| 一级总成供应商名称 | `object(str)` | `VARCHAR(255)` | 否 | 一级总成供应商元数据 | 可选索引 |
| 一级总成供应商代码 | `object(str)` | `VARCHAR(64)` | 否 | 一级总成供应商编码 | 可选索引 |
| 一级总成成本 | `float64` | `DECIMAL(18,4)` | 否 | 总成标准/原始成本，用于拆分件联动分析 | 否 |
| 源日期列 | `object(str)` | `TEXT` | 否 | 原始输入日期文本列，列名不固定；系统使用它解析出 `monitor_date` | 一般不建议作为正式核心字段 |

### 1.3 输入字段标准化映射

当前系统接收外部 CSV / Excel / JSON 时，会先通过字段映射统一成中文标准列。主要映射如下：

| 外部字段 | 标准字段 |
|---|---|
| `partId` / `materialCode` / `materialId` / `part_id` / `material_code` | `物料编码` |
| `partName` / `materialName` / `part_name` / `material_name` | `物料名称` |
| `vehicleSeries` / `vehicle_series` / `carModel` / `car_model` | `适用车系` |
| `shortName` / `short_name` / `partAlias` / `part_alias` | `备件简称` |
| `factory` / `plant` / `plantCode` / `plant_code` | `工厂` |
| `price` / `cost` / `unitPrice` / `unit_price` | `价格` / `成本` / `单价` |
| `validDate` / `effectiveDate` / `priceDate` 及其下划线变体 | `价格有效于` |
| `firstLevelAssyPartNo` / `assyPartNo` 及其下划线变体 | `一级总成料号` |
| `firstLevelAssyDesc` / `assyDesc` 及其下划线变体 | `一级总成品名描述` |
| `firstLevelAssySupplierName` / `assySupplierName` 及其下划线变体 | `一级总成供应商名称` |
| `firstLevelAssySupplierCode` / `assySupplierCode` 及其下划线变体 | `一级总成供应商代码` |
| `firstLevelAssyCost` / `assyCost` 及其下划线变体 | `一级总成成本` |

## 2. 异常检测结果表

建议将算法输出结果落为 `cost_anomaly_results` 表，用于支撑 UI、导出、专家标注和 Skills 生成。

### 2.1 表级建议

| 项目 | 建议 |
|---|---|
| 表名 | `cost_anomaly_results` |
| 主键 | `_record_key` |
| 常用索引 | `备件简称`、`status`、`物料编码`、`工厂`、`价格有效于` |
| 关联方式 | `_record_key` 对应专家标注表 `record_key` |

### 2.2 字段定义

| 字段 | 当前 Pandas 类型 | 建议 SQL 类型 | 必填 | 说明 | 主键/唯一/索引建议 |
|---|---|---|---|---|---|
| _record_key | `object(str)` | `VARCHAR(160)` | 是 | 由 `物料编码_工厂_价格有效于_实际成本` 拼接而成的逻辑唯一键 | 主键 / 唯一键 |
| 物料编码 | `object(str)` | `VARCHAR(64)` | 是 | 来自核心成本表 | 索引 |
| 物料名称 | `object(str)` | `VARCHAR(255)` | 否 | 来自核心成本表 | 否 |
| 适用车系 | `object(str)` | `VARCHAR(128)` | 否 | 来自核心成本表 | 索引 |
| 工厂 | `object(str)` | `VARCHAR(64)` | 否 | 来自核心成本表 | 索引 |
| 备件简称 | `object(str)` | `VARCHAR(128)` | 是 | 异常检测主分组维度 | 索引 |
| 实际成本 | `float64` | `DECIMAL(18,4)` | 是 | 原始成本值 | 否 |
| 价格有效于 | `datetime64[ns]` | `DATE` 或 `TIMESTAMP` | 是 | 由 `monitor_date` 重命名而来 | 索引 |
| 样本量 | `int64` | `INT` | 是 | 当前备件简称下的样本数量 | 否 |
| 预测值 | `float64` | `DECIMAL(18,4)` | 是 | 基准合理价 | 否 |
| 合理下限 | `float64` | `DECIMAL(18,4)` | 是 | 邻居圈左边界 | 否 |
| 合理上限 | `float64` | `DECIMAL(18,4)` | 是 | 邻居圈右边界 | 否 |
| 偏离数值 | `float64` | `DECIMAL(18,4)` | 是 | `实际成本 - 预测值` | 否 |
| 偏离比例 | `float64` / nullable | `DECIMAL(10,6)` | 否 | `偏离数值 / 预测值` | 否 |
| status | `object(str)` | `VARCHAR(64)` | 是 | `正常` / `异常偏高` / `异常偏低` / `严重异常偏低` 等状态 | 索引 |
| 专家校准 | `object(str)` | `VARCHAR(8)` | 否 | 仅加权模式输出，当前为空串或 `✅` | 否 |
| 判定依据 | `object(str)` | `VARCHAR(64)` | 否 | 仅加权模式输出，如 `默认算法` / `技能书校验` | 否 |

## 3. 专家标注数据结构

当前结构来自 `user_feedback.csv`，建议正式数据库表名为 `expert_feedback`。

### 3.1 表级建议

| 项目 | 建议 |
|---|---|
| 表名 | `expert_feedback` |
| 主键 | `record_key` |
| 常用索引 | `label`、`labeled_at` |
| 与核心数据关联 | 通过 `record_key = _record_key` 关联异常检测结果 |

### 3.2 字段定义

| 字段 | 当前类型 | 建议 SQL 类型 | 必填 | 说明 | 主键/唯一/索引建议 |
|---|---|---|---|---|---|
| record_key | `CSV string` | `VARCHAR(160)` | 是 | 与异常检测结果表 `_record_key` 一一对应 | 主键 / 唯一键 |
| label | `CSV string` | `VARCHAR(32)` | 是 | 当前 UI 实际写入 `正常`，结构上可扩展为更多标注枚举 | 索引 |
| labeled_at | `ISO datetime string` | `TIMESTAMP` | 是 | 标注写入时间 | 索引 |

### 3.3 与核心数据的关联规则

| 关联项 | 当前实现 |
|---|---|
| 关联主键 | `record_key = _record_key` |
| 关联来源 | 异常检测结果表 |
| 关系类型 | 逻辑上一对一：一条异常检测记录对应一条最新专家标注 |
| 建议做法 | 保留 `record_key` 的同时，在数据库中额外拆出 `物料编码`、`工厂`、`价格有效于`、`实际成本` 四列，便于审计和 SQL Join |

### 3.4 record_key 生成规则

```text
record_key = 物料编码 + "_" + 工厂 + "_" + 价格有效于(YYYY-MM-DD) + "_" + 实际成本(保留4位小数)
```

示例：

```text
16709144-00_CSK0_2023-11-27_4954.3500
```

## 4. Skills 技能书结构

当前运行时持久化文件为 `skills_active.json`，建议数据库可拆为 `skills_snapshot` 与 `skills_items` 两层。

### 4.1 顶层结构

| JSON 路径 | 当前类型 | 说明 | 主键/唯一/索引建议 |
|---|---|---|---|
| version | `string` | 文件结构版本，当前为 `1.0` | 否 |
| saved_at | `string(datetime ISO)` | 本地 Skills 保存时间 | 可作快照索引 |
| global_sigma | `number` | 全局 σ 参数 | 否 |
| global_weight | `integer` | 全局偏置权重 | 否 |
| skills | `array<object>` | Skills 条目数组 | 否 |

### 4.2 单个 skills[] 条目结构

| JSON 路径 | 当前类型 | 说明 | 主键/唯一/索引建议 |
|---|---|---|---|
| skills[].备件简称 | `string` | 条目主维度 | 在单次快照内可视为唯一键，索引 |
| skills[].适用算法 | `string` | 当前固定为 `KDE+KNN+Elbow 密度连接异常检测` | 否 |
| skills[].当前σ参数 | `number` | 当前条目的 σ 参数 | 否 |
| skills[].偏置权重 | `integer` | 当前条目的专家偏置权重 | 否 |
| skills[].本组专家标注数 | `integer` | 当前备件简称下专家标注数 | 否 |
| skills[].经验对齐率 | `number` 或 `string` | 有标注时是比例，无标注时为 `N/A` | 否 |
| skills[].数据结构分布描述 | `object` | 成本分布统计对象 | 否 |
| skills[].成本合理区间边界 | `object` | 预测值与合理上下限 | 否 |
| skills[].异常统计 | `object` | 当前备件简称下的异常统计 | 否 |

### 4.3 数据结构分布描述子结构

| JSON 路径 | 当前类型 | 说明 |
|---|---|---|
| skills[].数据结构分布描述.样本量 | `integer` | 样本数量 |
| skills[].数据结构分布描述.均值 | `number` | 成本均值 |
| skills[].数据结构分布描述.标准差 | `number` | 成本标准差 |
| skills[].数据结构分布描述.中位数 | `number` | 成本中位数 |
| skills[].数据结构分布描述.最小值 | `number` | 最小成本 |
| skills[].数据结构分布描述.最大值 | `number` | 最大成本 |
| skills[].数据结构分布描述.偏度 | `number` | 当样本量大于 2 时输出 |

### 4.4 成本合理区间边界子结构

| JSON 路径 | 当前类型 | 说明 |
|---|---|---|
| skills[].成本合理区间边界.预测值 | `number` | 基准合理价 |
| skills[].成本合理区间边界.合理下限 | `number` | 邻居圈左边界 |
| skills[].成本合理区间边界.合理上限 | `number` | 邻居圈右边界 |

### 4.5 异常统计子结构

| JSON 路径 | 当前类型 | 说明 |
|---|---|---|
| skills[].异常统计.正常 | `integer` | 正常记录数 |
| skills[].异常统计.异常偏高 | `integer` | 异常偏高记录数 |
| skills[].异常统计.异常偏低 | `integer` | 异常偏低与严重异常偏低合并数量 |

### 4.6 运行时结构与导出结构差异

| 项目 | 运行时持久化 `skills_active.json` | 导出 JSON |
|---|---|---|
| 时间字段 | `saved_at` | `generated_at` |
| 全局参数 | 有 `global_sigma`、`global_weight` | 无 |
| 数量字段 | 无 | 有 `skills_count` |
| 主体数据 | `skills` | `skills` |

## 5. 拆分件逻辑结构

当前拆分件成本监控逻辑基于“一级总成 - 子零件”的父子关系展开。

### 5.1 关系定义

| 角色 | 字段 | 说明 | 索引建议 |
|---|---|---|---|
| 一级总成主键 | `一级总成料号` | 父级总成连接字段 | 索引 |
| 子零件主键 | `物料编码` | 子零件编码 | 索引 |
| 时间维度 | `monitor_date` | 用于识别每个子零件的最新成本记录 | 复合索引 |
| 子零件成本 | `成本金额` / 动态 `price_col` | 子零件实际成本 | 否 |

### 5.2 原始关系表建议

建议正式数据库将“总成 - 子零件关系”保留在核心成本事实表中，或视业务需要拆出 `assembly_subpart_relation` 视图/表。

| 字段 | 当前 Pandas 类型 | 建议 SQL 类型 | 必填 | 说明 | 主键/唯一/索引建议 |
|---|---|---|---|---|---|
| 一级总成料号 | `object(str)` | `VARCHAR(64)` | 是 | 父级总成连接字段 | 索引 |
| 物料编码 | `object(str)` | `VARCHAR(64)` | 是 | 子零件编码 | 索引 |
| monitor_date | `datetime64[ns]` | `DATE` 或 `TIMESTAMP` | 是 | 用于选取子零件最新成本 | 复合索引 |
| 成本金额 | `float64` | `DECIMAL(18,4)` | 是 | 子零件当前成本 | 否 |
| 一级总成品名描述 | `object(str)` | `VARCHAR(255)` | 否 | 总成描述 | 否 |
| 一级总成供应商名称 | `object(str)` | `VARCHAR(255)` | 否 | 总成供应商名称 | 可选索引 |
| 一级总成供应商代码 | `object(str)` | `VARCHAR(64)` | 否 | 总成供应商代码 | 可选索引 |
| 一级总成成本 | `float64` | `DECIMAL(18,4)` | 否 | 总成原始成本 | 否 |

### 5.3 拆分件汇总结果结构

当前 `analyze_subpart_costs()` 输出建议落表为 `assembly_cost_audit`。

| 字段 | 当前 Pandas 类型 | 建议 SQL 类型 | 说明 | 主键/唯一/索引建议 |
|---|---|---|---|---|
| 一级总成料号 | `object(str)` | `VARCHAR(64)` | 汇总主键 | 主键 / 唯一键 |
| 一级总成品名描述 | `object(str)` | `VARCHAR(255)` | 总成描述 | 否 |
| 一级总成供应商名称 | `object(str)` | `VARCHAR(255)` | 总成供应商名称 | 否 |
| 一级总成供应商代码 | `object(str)` | `VARCHAR(64)` | 总成供应商代码 | 索引 |
| 一级总成成本 | `float64` | `DECIMAL(18,4)` | 原始总成成本 | 否 |
| 子零件数量 | `int64` | `INT` | 最新子零件数量 | 否 |
| 子零件加权总和 | `float64` | `DECIMAL(18,4)` | 当前实现为各子零件最新成本求和 | 否 |
| 测算总成成本 | `float64` | `DECIMAL(18,4)` | 当前实现为 `子零件加权总和 * 1.2` | 否 |
| 测算比值 | `float64` | `DECIMAL(10,4)` | `测算总成成本 / 一级总成成本` | 否 |
| 结论状态 | `object(str)` | `VARCHAR(32)` | `正常` / `异常` | 索引 |

### 5.4 子零件明细结构

当前 `get_subpart_detail()` 输出建议作为明细视图或 `assembly_subpart_latest_costs` 表。

| 字段 | 当前 Pandas 类型 | 建议 SQL 类型 | 说明 | 主键/唯一/索引建议 |
|---|---|---|---|---|
| 物料编码 | `object(str)` | `VARCHAR(64)` | 子零件编码 | 唯一键候选 |
| 物料名称 | `object(str)` | `VARCHAR(255)` | 子零件名称 | 否 |
| 备件简称 | `object(str)` | `VARCHAR(128)` | 子零件简称 | 索引 |
| 工厂 | `object(str)` | `VARCHAR(64)` | 子零件所属工厂 | 索引 |
| 子零件成本 | `float64` | `DECIMAL(18,4)` | 最新成本 | 否 |
| 价格有效于 | `datetime64[ns]` | `DATE` 或 `TIMESTAMP` | 最新成本日期 | 索引 |

## 6. 推荐数据库建模关系总览

### 6.1 推荐核心表

| 表名 | 作用 | 主键 |
|---|---|---|
| `core_cost_records` | 原始标准化后的成本事实表 | `cost_record_id` |
| `cost_anomaly_results` | 异常检测结果表 | `_record_key` |
| `expert_feedback` | 专家标注表 | `record_key` |
| `skills_snapshot` | Skills 顶层快照表 | `snapshot_id` |
| `skills_items` | Skills 每个备件简称的明细表 | `snapshot_id + 备件简称` |
| `assembly_cost_audit` | 一级总成拆分件汇总结果 | `一级总成料号` |

### 6.2 推荐关联关系

| 左表 | 右表 | 关联字段 | 关系说明 |
|---|---|---|---|
| `core_cost_records` | `cost_anomaly_results` | `物料编码 + 工厂 + monitor_date + 成本金额` 对应 `_record_key` 组成字段 | 原始成本记录生成异常检测结果 |
| `cost_anomaly_results` | `expert_feedback` | `_record_key = record_key` | 异常检测结果与专家标注一一关联 |
| `cost_anomaly_results` | `skills_items` | `备件简称` | Skills 按备件简称汇总 |
| `core_cost_records` | `assembly_cost_audit` | `一级总成料号` | 拆分件联动分析 |

## 7. 部署建模建议

| 建议项 | 建议内容 |
|---|---|
| 主键策略 | 原始事实表用独立主键，算法结果表保留 `_record_key` 作为业务唯一键 |
| 时间字段 | 优先使用 `DATE`；若后续需要保留更细粒度，可升级为 `TIMESTAMP` |
| 金额字段 | 统一使用 `DECIMAL(18,4)`，避免浮点误差 |
| 文本字段 | 编码统一 `utf8mb4`，兼容中文字段和值 |
| 索引策略 | 优先给高频筛选维度加索引：`物料编码`、`备件简称`、`适用车系`、`工厂`、`一级总成料号` |
| 审计性 | `expert_feedback` 和 `skills_snapshot` 建议保留历史版本，不要只覆盖最新值 |

