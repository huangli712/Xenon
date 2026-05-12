# 实施方案模板

> 本文档定义了所有 WxTy.md 实施方案文件的统一格式。
> 每个 task 文件必须严格遵循此模板。

---

## 文件命名

```
build/W{wave}T{task}.md
```

示例: `build/W2T4.md` 表示 Wave 2 的 Task 4。

---

## 模板内容

```markdown
# 实施方案: W{x}T{y}

## 任务概述

| 字段 | 值 |
|------|-----|
| Wave | W{x} — {Wave名称} |
| Task | W{x}T{y} — {任务简述} |
| 目标文件 | `{文件路径}` |
| 前置依赖 | {依赖列表} |
| 设计文档 | {设计文档引用} |
| 预估工时 | {时间估计} |

## 目标

{从 SUMMARY.md 复制的 Goal 描述，可以展开解释}

## 前置准备

{列举此任务开始前必须完成的事项}

## 实施步骤

### Step 1: {步骤名称}

{详细说明}

```rust
// 代码示例或伪代码
```

### Step 2: {步骤名称}

...

### Step N: 单元测试

> **强制要求**:
> - 本步骤必须包含一个或者多个单元测试。
> - 测试函数名称和测试内容必须参考对应设计文档的 **「测试计划」** 一节。
> - 编译检查与Lint检查不等同于单元测试。
> - 如果测试没有通过需要修复直至测试通过。
> - **无测试 = 任务未完成。**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // 测试函数命名: test_<场景描述>
    // 测试内容: 覆盖正常路径、边界情况、错误路径
    #[test]
    fn test_<scenario_name>() {
        // 参考设计文档「测试计划」节的具体测试用例
    }
}
```

#### 等效验证条款（仅适用于非 Rust 源码类目标文件）

当 task 的「目标文件」不是 Rust 源码（`.rs`）而是配置/工程类文件时——例如
`Cargo.toml`、`rustfmt.toml`、`.clippy.toml`、`.github/workflows/*.yml`、
`*.md`、`*.txt` 等——`#[test]` 单元测试无法直接施加于目标文件。此类 task
必须以 **「等效验证脚本」** 满足验收门禁，规则如下：

1. **不得以"配置文件无法测试"为由跳过本步骤**。
2. 必须提供一段或多段可执行的验证脚本（bash / python / cargo 子命令），且：
   - **任何检查失败必须以非零退出码返回**——禁止形如
     `grep ... && echo PASS || echo FAIL` 这种失败仍返回 0 的写法。
   - 推荐使用 `set -euo pipefail` 启动 bash 脚本；Python 检查使用 `assert`。
3. 验证脚本必须**至少覆盖**目标文件的以下检查面：
   - 语法/格式可解析（如 `cargo verify-project`、`tomllib.loads`、
     `yaml.safe_load`）；
   - 与所引设计文档章节**关键字段一一对照**（覆盖率不得弱于设计文档列出的项）；
   - 工具链可消费配置（如 `cargo clippy` 能读取 `.clippy.toml`）。
4. 「验证方式」一节必须显式列出运行等效验证脚本的命令，并纳入验收门禁。
5. 若 task 跨越 Rust 源码 + 配置文件（如同时修改 `src/foo.rs` 和 `Cargo.toml`），
   Rust 源码部分仍须包含 `#[test]` 单元测试，配置文件部分仍须包含等效验证脚本，
   二者并列存在，不得互相替代。

```bash
# 等效验证脚本示例（bash）
set -euo pipefail

# 1. 语法可解析
cargo verify-project

# 2. 与设计文档字段对照
python3 - <<'PY'
from pathlib import Path
import tomllib
config = tomllib.loads(Path("Cargo.toml").read_text())
assert config["package"]["edition"] == "2024", "edition must be 2024"
assert config["package"]["rust-version"] == "1.85", "MSRV must be 1.85"
PY

# 3. 工具链可消费
cargo metadata --format-version=1 > /dev/null
```

```python
# 等效验证脚本示例（python，需 3.11+ 提供 tomllib）
import sys, tomllib
from pathlib import Path

config = tomllib.loads(Path("rustfmt.toml").read_text())
expected = {"edition": "2024", "tab_spaces": 4, "max_width": 100}
for key, value in expected.items():
    assert config.get(key) == value, f"{key} must be {value!r}, got {config.get(key)!r}"
print("OK", file=sys.stderr)
```

> **注意**：等效验证条款是对「无测试 = 任务未完成」原则的**等效履行**，**不是豁免**。
> 验证脚本本身必须可执行、可失败、可纳入 CI。

## 关键设计决策

{列出此实现中的重要设计选择和权衡}

## 验证方式

{如何验证此 task 正确完成}

1. 编译检查: `cargo check`
2. Lint 检查: `cargo clippy`
3. 测试验证: `cargo test ...`
4. 手动验证: {具体方法}

## 注意事项

{陷阱、边界情况、与其它 task 的交互}
```

---

## 编委要求

1. **代码示例必须可编译**（标注依赖哪些类型/ trait 已存在）
2. **实施步骤必须原子化**，每步一个独立动作
3. **实施步骤最后一个 Step 必须为单元测试或等效验证**：
   - 目标文件为 Rust 源码（`.rs`）时，必须包含 `#[test]` 单元测试，测试函数和内容参考设计文档「测试计划」节；
   - 目标文件为非 Rust 配置/工程文件（如 `Cargo.toml`、`rustfmt.toml`、`.clippy.toml`、`.yml`、`.md` 等）时，必须按本模板「等效验证条款」提供可执行、可失败的验证脚本。
4. **无测试步骤（或无等效验证步骤）的方案视为未完成**
5. **验证方式必须可执行且具备失败语义**：
   - 所有 shell 检查命令在条件不满足时必须以非零退出码返回；
   - 禁止使用 `cmd && echo PASS || echo FAIL` 这种失败仍返回 0 的反模式。
6. **设计决策必须引用设计文档**中的具体章节
7. **注意事项必须覆盖**边界情况、错误处理、性能考量
8. **「任务概述」表头的「设计文档」字段必须精确到章节号**（如 `01-architecture.md §4`）。允许 task 表头列出比 SUMMARY.md 同行 Design Docs 字段**更细的章节子号**（例如 SUMMARY 写 `00-coding §12`，task 表头可扩写为 `00-coding.md §12, §3.2`），但**不得新增 SUMMARY 未列出的设计文档**，也**不得遗漏 SUMMARY 列出的章节**
