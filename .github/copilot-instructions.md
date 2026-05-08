---
name: fullstack-code-review
description: "Systematically audit full-stack projects for architecture flaws, runtime bugs, security vulnerabilities, data-layer anti-patterns, caching pitfalls, API design issues, and frontend optimization. Use when: code review before release; onboarding legacy codebase; sprint tech-debt triage; pre-deployment checklist. Covers Python, JS/TS, Streamlit, React, Vue, Django, Flask, FastAPI, Next.js."
---

# Full-Stack Code Review & Optimization

基于真实项目审计经验提炼的通用全栈代码审查框架。

**优先级标注**：P0 = 当日必修 · P1 = 本迭代修 · P2 = 下迭代 · P3 = 有空再做

## When to Use

- 发版前 code review
- 接手陌生仓库时的全面审计
- Sprint 回顾中梳理技术债
- CI/CD 准入门禁的检查项来源

## Procedure

```
Step 1 → 架构层：分层、API 规范、数据流、状态管理
Step 2 → 安全层：XSS / 注入 / 依赖 / Secret Scanning
Step 3 → 数据层：类型安全 / 边界值 / 缓存一致性
Step 4 → 前端层：导航 / 表格 / 图表 / 交互
Step 5 → 测试层：覆盖率 / 测试数据完整性
Step 6 → 性能层：数据处理 / 内存泄漏 / 渲染
Step 7 → 维护层：代码组织 / 错误处理 / 依赖
Step 8 → 部署层：环境兼容 / CI/CD / 容器化
```

逐层扫描，先修 P0 → P1，安全和正确性优先于体验优化。

---

## 一、架构层 (Architecture)

### 1.1 分层与职责

| 检查项 | 说明 | 反面案例 | 修复方向 |
|--------|------|---------|----------|
| 前后端是否明确分离 | 数据处理逻辑不应出现在视图层 | UI 文件中直接操作 DataFrame / SQL | 提取到独立 service / processor 模块 |
| 业务逻辑是否可独立测试 | 核心算法能脱离框架单独跑 | 算法函数内部调用 `st.cache_data` 等框架 API | 缓存装饰器放调用侧，算法函数保持纯净 |
| 路由/页面管理是否集中 | 多页面路由应有统一入口 | `if/elif` 链条 + `session_state` 手动模拟路由 | 使用框架原生路由（Streamlit `st.navigation`、React Router、Vue Router） |
| 配置是否外部化 | 硬编码路径、列名、阈值 | 魔术字符串散落全文件 | 集中到 `config.py` / `.env` / `settings.yaml` |

### 1.2 API 设计规范

| P级 | 检查项 | 说明 |
|-----|--------|------|
| P1 | **RESTful / GraphQL 规范** | 端点命名是否遵循资源导向（`/api/v1/materials`），HTTP 方法是否语义正确（GET 读 / POST 写 / DELETE 删），避免 `POST /doSomething` 式 RPC 命名 |
| P1 | **版本化** | API 路径中是否体现版本号（`/v1/`、`/v2/`），或使用 Header 版本协商（`Accept: application/vnd.api.v2+json`），确保旧客户端不被破坏 |
| P2 | **统一响应格式** | 是否有标准的 `{ code, data, message }` 或 RFC 7807 Problem Details 响应体结构；错误码是否有文档 |
| P2 | **分页/过滤约定** | 列表接口是否有统一的分页参数（`page`/`page_size` 或 `cursor`/`limit`），以及排序、筛选的查询参数规范 |
| P3 | **幂等性** | PUT / DELETE 是否幂等；POST 创建是否有去重 Token |

### 1.3 数据流与状态管理

| P级 | 检查项 | 说明 |
|-----|--------|------|
| P0 | **关键元数据是否跟随数据本体** | 对后续计算至关重要的值（如「价格列名」），绝不能放在可能静默丢失的位置（DataFrame `_attrs`、全局变量、闭包）。必须作为函数返回值或独立状态字段显式传递 |
| P1 | **缓存键是否充分** | `@cache(key=folder_path)` 只看路径字符串——文件内容变化但路径不变时缓存不失效。应纳入文件修改时间戳或内容哈希 |
| P1 | **状态持久化** | 用户配置只存内存 session → 刷新即丢失。应持久化到本地文件或数据库 |
| P2 | **大对象在 session 中的生命周期** | 完整 DataFrame 存 `session_state`，多标签共享时内存翻倍。考虑 `@st.cache_resource` 或数据库连接池 |

### 常见架构反模式

```
❌ 单文件巨石（God File）
   app.py 同时包含路由、CSS、数据加载、图表渲染、导出。
   → 按页面/功能拆分模块，每个文件 < 300 行。

❌ 装饰器双重叠加
   processor.py 函数有 @st.cache_data，app.py wrapper 又包一层。
   → 缓存只在调用侧做一次；核心函数不依赖框架。

❌ 私有/实验性 API 传递关键数据
   用 DataFrame._attrs → concat/copy/序列化后丢失。
   → 显式返回 tuple (df, metadata) 或封装 dataclass。
```

---

## 二、安全层 (Security)

### 2.1 XSS / HTML 注入（P0）

任何用户可控数据（含 Excel / CSV / DB 字段）直接拼入 HTML 字符串并通过 `innerHTML` / `unsafe_allow_html` / `v-html` / `dangerouslySetInnerHTML` 渲染 → XSS。

```python
import html

def escape(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return html.escape(str(value), quote=True)

f'<td>{escape(user_data)}</td>'       # ✅
f'<td>{user_data}</td>'                # ❌ XSS
```

| 框架 | 危险 API | 安全替代 |
|------|----------|----------|
| Streamlit | `st.markdown(html, unsafe_allow_html=True)` | 数据部分用 `html.escape()`；结构 HTML 可保留 |
| React | `dangerouslySetInnerHTML` | 用 JSX 渲染；必须用时先 DOMPurify |
| Vue | `v-html` | 用 `{{ }}` 插值（自动转义） |
| Django | `\|safe` / `mark_safe()` | 默认模板自动转义；仅对已确认安全内容用 `\|safe` |
| Express/EJS | `<%- raw %>` | 用 `<%= escaped %>` |

### 2.2 路径遍历（P1）

```python
# ❌ 用户输入直接拼路径
files = glob.glob(os.path.join(user_input, "*.xlsx"))

# ✅ 白名单 + 路径规范化
import pathlib
base = pathlib.Path("/allowed/data/root").resolve()
target = (base / user_input).resolve()
if not str(target).startswith(str(base)):
    raise ValueError("路径越界")
```

### 2.3 依赖安全（P1）

| 检查项 | 说明 |
|--------|------|
| `requirements.txt` 是否列全 | 传递依赖（如 numpy）会被 pandas 拉入，但 scikit-learn 等可选依赖不列 → 新环境运行时报错 |
| 是否有版本锁定 | 不锁版本 → 大版本升级引入 breaking change |
| 依赖审计 | 定期跑 `pip-audit` / `npm audit` / `snyk test` 检查 CVE |

### 2.4 Secret Scanning（P1）

| 检查项 | 说明 |
|--------|------|
| CI 集成密钥扫描 | 在 CI pipeline 中加入 [gitleaks](https://github.com/gitleaks/gitleaks) 或 [trufflehog](https://github.com/trufflesecurity/trufflehog) 作为 pre-commit hook 和 PR 检查 |
| `.env` 不入库 | `.env`、`*.pem`、`*_secret*` 必须写入 `.gitignore`；历史泄漏需 `git filter-repo` 清理 |
| GitHub Secret Scanning | 启用 GitHub Advanced Security 的 push protection，在 push 阶段拦截已知格式的 token/key |
| 密钥轮换 | 一旦发现泄漏，立即轮换而非仅从代码中删除 |

### 2.5 其他安全要点

| P级 | 项 | 说明 |
|-----|---|------|
| P1 | 异常不要静默吞掉 | `except Exception: continue` 隐藏失败原因，应至少 `logging.warning` |
| P1 | 敏感信息不入日志 | 打印完整文件路径/数据预览时注意脱敏 |
| P2 | CSRF / CORS | Web API 应配置白名单 Origin |
| P2 | 文件上传限制 | 限制大小、校验 MIME 类型 |
| P2 | Rate Limiting | 公开 API 应有请求频率限制，防止暴力破解和滥用 |

---

## 三、数据层 (Data Layer)

### 3.1 类型安全与边界值

| P级 | 检查项 | 说明 | 修复 |
|-----|--------|------|------|
| P0 | 除零保护 | `a / b if b else 0` → `0.0` 时 `if 0.0` 为 False，语义不明 | 显式 `if b != 0` 或 `pd.Series.replace(0, pd.NA)` |
| P1 | NaN 传播 | Pandas `NaN != NaN` | 始终用 `pd.isna()` 检查 |
| P1 | 日期解析健壮性 | 正则 `\d{4}-\d{2}-\d{2}` 会匹配 `9999-99-99` | 加 `pd.to_datetime(..., errors='coerce')` 后续过滤 |
| P2 | 大数精度 | 浮点累加、货币精度丢失 | 关键金额用 `Decimal` 或整数分为单位 |

### 3.2 数据加载与合并

```
✅ 优秀实践
   - 每个文件独立校验 schema（必要列是否存在）
   - concat 后检查列名冲突
   - 返回加载摘要（成功/失败文件数、失败原因列表）

❌ 常见问题
   - schema 不一致时静默丢列
   - 编码探测只尝试 utf-8 / gbk，遗漏 gb2312 / latin-1
   - glob 不递归子目录
```

### 3.3 缓存策略

| 场景 | 推荐策略 |
|------|----------|
| 静态配置/模型加载 | `@st.cache_resource` / 模块级单例 |
| 纯函数计算 | `@st.cache_data` / `@functools.lru_cache` |
| 外部数据源（文件/API/DB） | 缓存键纳入时间戳/哈希；提供手动刷新按钮 |
| 大型 DataFrame | Arrow / Parquet 中间格式减少序列化开销 |

**反模式 — 双重缓存**：

```python
# processor.py
@st.cache_data                    # ← 第一层
def detect_anomalies(df, col): ...

# app.py
@st.cache_data                    # ← 第二层（冗余、内存浪费、一致性风险）
def cached_anomaly(df, col):
    return processor.detect_anomalies(df, col)
```

---

## 四、前端层 (Frontend)

### 4.1 导航与路由

| P级 | 问题 | 优化方向 |
|-----|------|----------|
| P2 | `st.button` + `session_state` 模拟路由 | 改为 `st.radio` / `st.selectbox`（Streamlit）或原生路由（React Router / Vue Router） |
| P2 | 页面切换无 loading 过渡态 | 页面入口包裹 `st.spinner` / React `<Suspense>` |
| P3 | 手动分页只能上一页/下一页 | 增加页码输入框或跳转 |

### 4.2 CSS 与样式管理

| P级 | 问题 | 优化方向 |
|-----|------|----------|
| P2 | 每次 rerun 重复注入 `<style>` | 提取到 `.css` 文件；`st.html` (≥1.33) 或 `@st.cache_resource` 注入一次 |
| P2 | 大量 `!important` 覆盖框架样式 | 用更精确的选择器替代 |
| P3 | 内联样式分散在 Python 字符串中 | 集中到 CSS class |

### 4.3 表格与数据展示

| P级 | 问题 | 优化方向 |
|-----|------|----------|
| P2 | 手写 HTML `<table>` 拼字符串渲染 | 轻量需求用 `st.dataframe` + `column_config`；复杂需求用 AG Grid |
| P2 | 手写表格失去排序/筛选/列宽调整 | 框架原生数据表组件自带 |
| P3 | rowspan 合并要手写 HTML | 封装为独立渲染函数并充分测试 |

### 4.4 图表与可视化

| P级 | 问题 | 优化方向 |
|-----|------|----------|
| P2 | 不同量级数据混入同一直方图 | 分组汇总视图（Top-N 柱图）或按类别分 facet |
| P3 | 图表无空状态提示 | 数据为空时显示占位图或友好提示 |
| P3 | 颜色方案硬编码 | 提取为主题配置，支持暗色模式 |

### 4.5 交互体验

| P级 | 问题 | 优化方向 |
|-----|------|----------|
| P1 | 数据同步后只有 toast，无预览 | 增加 `df.head()` 预览 + 列名/类型摘要 |
| P2 | tkinter 弹窗在无 GUI 环境崩溃 | 改为 `st.text_input` 或 `st.file_uploader` |
| P2 | 用户配置刷新即丢失 | 自动持久化到本地 JSON/YAML |
| P3 | 搜索无 debounce | 用 `st.form` 或 `on_change` 配合延迟触发 |

---

## 五、测试层 (Testability)

### 5.1 测试覆盖

| P级 | 检查项 | 说明 |
|-----|--------|------|
| P1 | **无任何测试文件** | 至少覆盖数据处理核心函数 |
| P1 | **测试数据不完整** | 生成的数据缺少关键列 → 无法覆盖核心功能 |
| P2 | **边界情况未覆盖** | 空文件夹、全空值 DataFrame、单行数据、全同值 |

### 5.2 推荐测试结构

```
tests/
├── test_process_dataframe.py    # 列检测、日期解析、缺列兜底
├── test_pivot_report.py         # 透视表生成、截断
├── test_trend_report.py         # 变动趋势计算
├── test_anomaly_detection.py    # 小样本回退、单值、KDE
├── test_html_rendering.py       # XSS 转义、空数据、rowspan
├── test_data_loading.py         # 编码探测、混合格式、路径异常
└── conftest.py                  # 共享 fixture
```

### 5.3 测试原则

1. 核心算法不依赖框架 → 可直接 pytest 调用
2. 每个公开函数至少一个 happy path + 一个 edge case
3. HTML 渲染测试需断言：数据已转义、空 DataFrame 返回占位提示、rowspan 与分组行数一致
4. 数据加载用 `tmp_path` fixture，不依赖硬编码路径

---

## 六、性能层 (Performance)

### 6.1 数据处理性能

| P级 | 问题 | 优化 |
|-----|------|------|
| P2 | `groupby` + `apply` 逐行 lambda | 向量化操作替代逐行调用 |
| P2 | 每个分组独立 fit KDE + KNN | 大数据集 O(n²)；可并行化或增量计算 |
| P3 | `df.copy()` 过多 | 只在必须修改时 copy，只读路径用 view |
| P3 | HTML 表格字符串拼接 | 大表格用 `io.StringIO` 或 `"".join(list)` |

### 6.2 内存泄漏（P2）

长期运行的 Python 服务（如 Streamlit / Flask / FastAPI 驻进程模式）需特别关注：

| 场景 | 说明 | 修复 |
|------|------|------|
| 全局 `list.append()` 不释放 | 日志列表、历史记录 list 随请求无限增长 | 改用 `collections.deque(maxlen=N)` 或定期清空；写磁盘/DB |
| 缓存无上限 | `@lru_cache` 不设 `maxsize` → 长时间运行后占满内存 | 显式设置 `maxsize`；使用 TTL 缓存（`cachetools.TTLCache`） |
| DataFrame 引用链 | `session_state` 存上一次的 DataFrame 副本 → 历史副本不被 GC | 更新时显式 `del st.session_state["old_key"]` |
| 循环引用 + `__del__` | 含 `__del__` 方法的对象形成循环引用 → GC 无法回收 | 使用 `weakref`；避免在 `__del__` 中引用其他对象 |
| 第三方 C 扩展 | numpy / pandas 底层 C 分配的内存 Python GC 不感知 | 用 `tracemalloc` 定位；大数组用完后显式 `del` + `gc.collect()` |

```python
# ❌ 全局列表无限增长
request_log = []
def handle_request(req):
    request_log.append(req)  # 进程不重启就永远不释放

# ✅ 有界容器 + 持久化
from collections import deque
request_log = deque(maxlen=1000)
```

### 6.3 前端渲染性能

| P级 | 问题 | 优化 |
|-----|------|------|
| P2 | 每次 rerun 重新注入相同 CSS | 用 `@st.cache_resource` 返回 CSS 字符串 |
| P2 | 大表格一次性渲染 HTML string | 分页 + 虚拟滚动（`st.dataframe` 原生支持） |
| P3 | Plotly 图表未关闭不需要的交互 | `fig.update_layout(dragmode=False)` |

---

## 七、维护层 (Maintainability)

### 7.1 代码组织

| 检查项 | 标准 |
|--------|------|
| 单文件行数 | < 300 行（不含空行和注释） |
| 函数长度 | < 50 行；超过则拆分 |
| 嵌套深度 | 最大 3 层 `if/for`；超过则提取 helper |
| 魔术数字/字符串 | 全部提取为命名常量 |

### 7.2 错误处理

```python
# ❌ 静默吞异常
try:
    raw_df = pd.read_csv(filename)
except Exception:
    continue

# ✅ 记录并汇报
import logging
logger = logging.getLogger(__name__)
errors = []
try:
    raw_df = pd.read_csv(filename)
except Exception as e:
    logger.warning("跳过文件 %s: %s", filename, e)
    errors.append((filename, str(e)))
    continue
if errors:
    st.warning(f"有 {len(errors)} 个文件读取失败")
```

### 7.3 依赖管理

```
# requirements.txt 应 pin 主版本
streamlit>=1.30,<2.0
pandas>=2.0,<3.0
plotly>=5.0,<6.0
openpyxl>=3.1
numpy>=1.24
scikit-learn>=1.3    # 可选功能标注或拆到 requirements-ml.txt
```

---

## 八、部署层 (Deployment)

| P级 | 检查项 | 说明 |
|-----|--------|------|
| P1 | GUI 依赖 (tkinter) | Docker / 云服务器上 `tk.Tk()` 崩溃。改用纯 Web 输入或检测环境后降级 |
| P2 | 无 Dockerfile | 部署环境不可复现 |
| P2 | 无 CI/CD 配置 | 缺少自动化测试、lint、打包流程 |
| P2 | 无 Secret Scanning in CI | 加入 gitleaks / trufflehog pre-commit hook |
| P3 | 无 `.gitignore` | `.venv`、`__pycache__`、临时数据可能被提交 |

---

## 九、审查清单 (Checklist)

### P0 — 必须当日修复
- [ ] 所有用户可控数据渲染到 HTML 前已转义
- [ ] 关键元数据通过显式返回值传递，不依赖私有/实验性 API
- [ ] 不存在未受保护的路径拼接、SQL 注入、命令注入

### P1 — 当前迭代修复
- [ ] `requirements.txt` 列出所有直接依赖并锁定主版本
- [ ] 缓存策略正确：无双重缓存；外部数据源缓存键含时间戳
- [ ] 异常不被静默吞掉；失败原因对用户可见或可追溯
- [ ] 核心函数有至少 1 个单元测试
- [ ] CI 中集成 Secret Scanning（gitleaks / trufflehog）
- [ ] API 端点遵循 RESTful 规范并有版本号

### P2 — 下次迭代
- [ ] 导航使用框架原生组件
- [ ] CSS 集中管理，不重复注入
- [ ] 用户配置持久化到磁盘
- [ ] 分页支持跳转
- [ ] 数据同步后提供预览
- [ ] 长期运行服务排查内存泄漏（全局容器、缓存无上限）

### P3 — 有空再做
- [ ] 图表支持暗色模式
- [ ] 搜索增加 debounce
- [ ] 提供 Dockerfile
- [ ] CI/CD 流水线

---

## 十、技术栈速查

### Streamlit

| 陷阱 | 说明 |
|------|------|
| `st.cache_data` 序列化丢失 `DataFrame._attrs` | 使用显式返回 tuple |
| 每个 widget 需唯一 `key` | 动态生成 key 不能重复 |
| `st.rerun()` 重新执行整个脚本 | 昂贵计算必须在 cache 保护下 |
| `unsafe_allow_html=True` | 只用于固定模板结构；数据部分必须转义 |
| `st.file_uploader` 有大小限制 | 默认 200MB，需配置 `server.maxUploadSize` |

### React / Next.js

| 陷阱 | 说明 |
|------|------|
| `dangerouslySetInnerHTML` | 必须先 DOMPurify；优先用 JSX |
| useEffect 依赖数组遗漏 | 导致 stale closure / 无限循环 |
| 大列表未虚拟化 | 使用 `react-window` / `react-virtuoso` |
| API 密钥暴露到客户端 | 只在 Server Component / API Route 中使用 |

### FastAPI

| 陷阱 | 说明 |
|------|------|
| async 路由中调用同步阻塞 IO | `open()` / `requests.get()` / `time.sleep()` 会阻塞整个事件循环 → 用 `aiofiles`、`httpx.AsyncClient`、`asyncio.sleep()`，或将同步代码包在 `run_in_executor()` 中 |
| 未配置 CORS middleware | 前端跨域请求被拒。需 `app.add_middleware(CORSMiddleware, allow_origins=[...])` |
| Pydantic model 与 ORM model 混用 | 直接返回 SQLAlchemy 对象 → 序列化失败或暴露内部字段。使用 `response_model` + 独立 Schema |
| 依赖注入中的数据库连接泄漏 | `Depends(get_db)` 如果 generator 未正确 `yield` + `finally close()` → 连接池耗尽 |
| `BackgroundTasks` 中异常静默丢失 | 后台任务抛异常不会返回给客户端。需在任务内部 try/except 并写日志 |

### Flask

| 陷阱 | 说明 |
|------|------|
| 全局变量在多 worker 间不共享 | gunicorn 多 worker 模式下 `global_dict` 各进程独立 → 用 Redis / DB 做共享状态 |
| `app.run(debug=True)` 上生产 | 暴露错误堆栈和自动重载。生产使用 gunicorn / waitress |
| 同步阻塞架构 | 默认 WSGI 同步 → 长耗时请求阻塞 worker。用 Celery 异步化或迁移至 async 框架 |
| `send_file` 路径未校验 | 用户输入拼文件路径 → 路径遍历。必须 `safe_join()` 或 `send_from_directory()` |
| `SECRET_KEY` 硬编码 | 应从环境变量读取：`app.secret_key = os.environ["SECRET_KEY"]` |

### Django

| 陷阱 | 说明 |
|------|------|
| `\|safe` / `Markup()` 滥用 | 只对确认安全内容使用 |
| ORM N+1 查询 | 使用 `select_related` / `prefetch_related` |
| `DEBUG=True` 上生产 | 暴露错误堆栈和环境变量 |
| CSRF 中间件被禁用 | 仅 API-only 时才考虑，且需用 JWT 等替代 |
