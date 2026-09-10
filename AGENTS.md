# AGENTS.md — Style-Agnostic WAM
参考docs/guideline.md
conda环境为style_agnostic_wam
装包用阿里/清华源,同时必须同步requirements.txt;

- 尽一切可能用国内源,如果小流量请求不通可以查看本地配置:local_network.md

## 单元测试纪律（硬性）

- 每次改动网络模块（src/sawvla/models/、src/sawvla/losses/）后**必须**运行全部单元测试：
  `python -m pytest tests/ -q`（conda 环境 style_agnostic_wam）。
- **全部通过才算完成**；测试清单与各用例覆盖作用见 docs/unittests.md，新增/改动测试须同步更新该文档。
- 有不通过的测试时，**禁止**为了让测试变绿而修改/删除/放宽测试断言或跳过条件；必须先向用户说明失败原因并申请，得到用户明确同意后才可以修改通过标准。