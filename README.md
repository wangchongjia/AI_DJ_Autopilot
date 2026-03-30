# AI_DJ_Autopilot · AI DJ Copilot（本地原型）

这是一个本地原型（不做推荐/建模），只做两件事：

1. 扫描并提取小型 K-pop 音乐库的音频特征（librosa）
2. 用 Streamlit 可视化：曲目表、筛选/排序、BPM/能量分布、BPM vs 能量散点图、CSV 导出与摘要统计

## 环境准备

- Python 3.11+
- 建议：macOS 上如果你加载 mp3 失败，通常需要安装 `ffmpeg`

安装依赖：

```bash
cd "/Users/wangchongjia/Desktop/AI-DJ-Autopilot"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 环境变量（推荐）

你可以直接导出环境变量，或复制 `.env.example` 为 `.env`（摄取脚本会尝试自动加载 `.env`）。

- **KPOP_SONGS_DIR**：要扫描的歌曲目录（递归）
- **AI_DJ_DB_PATH**：SQLite 数据库路径

示例：

```bash
cp .env.example .env
```

## 1) 数据摄取（扫描 + 特征提取 + 写入 SQLite）

默认扫描目录是项目内的 `Songs/kpop`（也就是当前工作区你已有的目录）。

你可以通过环境变量覆盖为你希望的目录（例如你说的 `/songs/kpop`）：

```bash
export KPOP_SONGS_DIR="/songs/kpop"
```

然后运行摄取脚本：

```bash
python scripts/ingest_kpop.py --db data/tracks.db
```

你也可以显式传入扫描目录：

```bash
python scripts/ingest_kpop.py --songs-dir "/songs/kpop" --db data/tracks.db
```

运行完成后，特征结果会保存到 `data/tracks.db`。

## 2) 启动可视化（Streamlit）

```bash
streamlit run app/streamlit_app.py
```

页面会从本地 SQLite 读取所有曲目并展示统计图。

## 3) 快速验证（Sanity Check）

摄取后你可以跑一个轻量验证脚本，快速查看分布与潜在异常（例如 BPM 缺失、BPM 过低/过高、超短音频等）：

```bash
python scripts/validate_features.py --db data/tracks.db
```

## 4) 本地清理 + 重训（不重新标注）

当你手动删除了坏音频（例如过短片段）后，可以直接清理 SQLite，并重建数据集+重训 baseline。

1) 备份数据库：

```bash
cp data/tracks.db data/tracks.backup.$(date +%Y%m%d_%H%M%S).db
```

2) Dry-run 查看将删除的轨道和标签（示例：删除 duration_sec < 90）：

```bash
python scripts/cleanup_tracks.py --db data/tracks.db --duration-lt 90 --dry-run
```

3) 执行实际清理：

```bash
python scripts/cleanup_tracks.py --db data/tracks.db --duration-lt 90
```

4) 可选：按精确 file_path 删除（可重复传参）：

```bash
python scripts/cleanup_tracks.py --db data/tracks.db \
  --file-path "Songs/kpop/bad_clip_1.mp3" \
  --file-path "Songs/kpop/bad_clip_2.mp3"
```

5) 重建 pair 数据集（训练用）：

```bash
python scripts/build_pair_dataset.py --db data/tracks.db --out-csv data/modeling/pair_dataset.csv
```

6) 重训 baseline（不新增模型、不要求重标注）：

```bash
python scripts/train_baseline_model.py \
  --db data/tracks.db \
  --dataset-csv data/modeling/pair_dataset.csv \
  --model-out data/modeling/baseline_logreg.joblib \
  --metrics-out data/modeling/baseline_metrics.json
```

## 5) 过渡感知特征（intro/outro）+ baseline 重训

在整轨特征之外，摄取会为每首歌计算 **outro（末尾约 N 秒）** 与 **intro（开头约 N 秒）** 的片段特征（默认 `N=15`，可用 `--segment-window` 调成约 10–20 秒）。更新代码后请先**重新摄取**，再重建 pair 数据集并重训 Logistic Regression；可用 `scripts/analyze_logreg_coefficients.py` 查看权重最大的正负特征。

```bash
python scripts/ingest_kpop.py --db data/tracks.db --segment-window 15
python scripts/build_pair_dataset.py --db data/tracks.db --out-csv data/modeling/pair_dataset.csv
python scripts/train_baseline_model.py \
  --db data/tracks.db \
  --dataset-csv data/modeling/pair_dataset.csv \
  --model-out data/modeling/baseline_logreg.joblib \
  --metrics-out data/modeling/baseline_metrics.json
python scripts/analyze_logreg_coefficients.py --model-path data/modeling/baseline_logreg.joblib --top 10
```

系数摘要默认写入 `data/modeling/logreg_coefficients.json`。

## 说明：关键/能量分数可靠性

- “estimated key（估计调性）”在原型阶段是启发式实现，准确率在不同歌曲上可能不理想（尤其是有明显混音/调性变化/短片段时）。
- “energy_score”是基于全库的 RMS 归一化得到的相对能量分数，适合做简单可视化和后续扩展，但不是绝对响度标准。
