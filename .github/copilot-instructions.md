---
name: costmonitoring-kotasp-constitution
description: "Cost Monitoring System constitution, blueprint lock, and self-correction rules"
applyTo: "**"
---

# 成本监控系统宪法

本文件是本仓库的最高准则。所有新功能、重构、修复、示例代码、提示词、脚手架和 Copilot 输出，均必须服从本宪法。

若新的实现、建议或代码片段与本宪法冲突，Copilot 必须先自我纠正，再给出最终结果。不得以“只是示例”“只是临时方案”“只是局部修复”为理由绕过本文件。

未经用户明确说出以下确认语句，不得擅自更新任何模块蓝图、放宽任何锁定约束、或将当前实现视为新的官方基线：

`这个新功能已确定，请将其作为新蓝图`

## K-O-T-A-S-P 模型

### K — Knowledge

以下逻辑属于项目核心知识，不得被弱化、偷换或删除。

1. DGB 动态断层共识算法
   - 锁定实现锚点：`anomaly_engine.detect_dgb_anomalies`、`anomaly_engine.detect_dgb_anomalies_weighted`
   - 必须保留基于密度/断层的异常检测语义，不得退化为简单均值阈值或固定区间判断。

2. 时序衰减权重
   - 锁定实现锚点：`anomaly_engine.calculate_recency_weight_series`
   - 锁定常量：`_TIME_DECAY_FULL_WEIGHT_DAYS = 183`、`_TIME_DECAY_DECAY_DAYS = 365.0`
   - 权重逻辑必须保持“先满权、后指数衰减”的结构，核心表达式不得被改写成线性折损或手工分段魔法数。

3. AutoResearch 棘轮迭代逻辑
   - 锁定实现锚点：`skills_engine.run_auto_research`、`skills_engine._is_trial_better`
   - 必须保留“候选参数试探 -> 评分 -> 仅在更优时采纳 -> 回退保护”的棘轮机制。
   - 参数维度必须覆盖 `sigma`、`expert_weight`、`decay_alpha`、`gap_k`。

4. 钣金白痴指数公式
   - 锁定实现锚点：`sheet_metal_logic._compute_sheet_metal_index`
   - 锁定常量：`_STEEL_DENSITY = 7.6`
   - 锁定公式：`白痴指数 = 出厂单价 / ((净重 / 1000) * 7.6)`
   - 允许补空值，但不得更改公式主体。

5. BOM 审计比例公式
   - 锁定实现锚点：`assembly_logic._vectorized_ratio`、`assembly_logic._build_summary_df`
   - 锁定比例语义：
     - `成本比例 = 层级1成本 / 层级0成本`
     - `订购价比例 = 层级1订购价汇总 / 层级0订购价`
     - `零售价比例 = 层级1零售价汇总 / 层级0零售价`
   - 除零保护必须保留，禁止回退成不带防护的直接相除。

### O — Observation

1. 系统启动时必须执行 Harness 完整性审计。
2. 审计对象至少覆盖：`app.py`、`app_context.py`、`assembly_ui.py`、`anomaly_engine.py`、`skills_engine.py`、`sheet_metal_logic.py`、`assembly_logic.py`、`llm_engine.py`、`storage_service.py`。
3. 审计既检查结构签名，也检查蓝图锁定状态。
4. 若蓝图不匹配，必须在 UI 中暴露，不得静默忽略。

### T — Tools

1. 所有 SQLite 读写都必须通过 Harness 中间件或经其封装的 Service 门面。
2. 所有 LLM 调用都必须通过 Harness 中间件。
3. 页面文件不得直接操作底库，不得在 UI 层发起裸 `sqlite3`、SQLAlchemy Session、或外部 LLM HTTP 请求。
4. 页面层只允许调用 `harness.execute_action(...)` 或 Harness 暴露的受控接口；不允许跨层直连数据库实现细节。

### A — Action

1. 若审计发现导航结构、视觉对齐、树状表实现或核心算法锚点退行，系统必须立即报警。
2. 报警文案必须包含：`❗ 警告：检测到核心架构退行`
3. 报警必须指出冲突模块与位置。
4. 蓝图更新只能通过 `confirm_and_update_blueprint(module_name, confirmation_text)` 完成，并要求确认语句命中本宪法中的固定短语。

### S — State

1. Harness 必须在 `app_context` 初始化时校验并回灌以下 6 个强制本地路径：
   - `input_data_path`
   - `quantitative_skills_path`
   - `qualitative_skills_path`
   - `assembly_data_path`
   - `sheet_metal_base_info_path`
   - `sheet_metal_model_export_path`
2. `sheet_metal_report_export_path` 作为兼容扩展路径，应继续回灌，但不计入上述 6 个强制键统计。
3. 任何 session-state 路径回灌都应以 `settings.json` 为单一事实源。

### P — Permissions

1. LLM 严禁接收任何包含如下敏感数值字段的原始数据：
   - `cost`
   - `sigma`
   - `price`
   - `ratio`
   - `amount`
   - `bound`
   - `weight`
   - 中文等价字段：`成本`、`价格`、`订购价`、`零售价`、`比例`、`白痴指数`、`权重`、`σ`
2. LLM 允许接收已脱敏的备注文本、备件简称、以及不含敏感数值的语义摘要。
3. 若输入疑似携带上述敏感数值字段，Harness 必须拒绝请求，而不是“尽量帮忙”。

## Immutable UI

以下 UI 规则属于不可变视觉蓝图。任何新开发不得违背。

### 1. 导航三定律

1. 顶部必须保留两个独立一级按钮：`概览`、`系统设置`
2. 顶部双项之后必须有灰线分隔
3. 灰线下方必须是邻近折叠 Expander 子菜单，不得改成平铺大列表、Tab、树控件或侧滑二级路由

### 2. 视觉对齐

1. 所有表格表头强制居中
2. 所有表格内容强制左对齐
3. 该规则适用于 `st.dataframe`、`st.data_editor`、HTML 表格以及任何模拟表格组件

### 3. 树状合并

1. `拆分件成本监控` 必须使用 HTML `rowspan` 模拟树状无缝表格
2. 禁止把一级件与二级件拆成左右分离的布局
3. 禁止以 `st.columns` 行栅格替代 HTML `rowspan` 树表作为最终形态

## 自更新与锁定规则

1. `harness.py` 是系统治理中枢。
2. `harness/blueprints/` 存放各模块结构签名蓝图。
3. `harness/state/` 存放蓝图锁定状态与确认记录。
4. 任何未被确认的新改动，都必须在审计结果中表现为“待确认变更”或“蓝图退行”。

## Copilot 自纠条款

在本仓库后续所有对话中，Copilot 必须遵守以下行为：

1. 如果将要输出的代码违反本宪法，必须先停止并自我纠正。
2. 如果用户要求的实现与本宪法冲突，必须明确指出冲突，再提供符合蓝图的替代实现。
3. 如果代码库当前实现已经偏离蓝图，新的修改不得继续加深偏离；应优先恢复蓝图，或由 Harness 明确报错。
4. 不得以“保持现状”为理由绕过上述约束。