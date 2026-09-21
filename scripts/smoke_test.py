"""端到端冒烟测试：算法正确性 + FastAPI 接口 + SQLite 持久化。

运行（需安装 requirements）：
    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.detection import analyze_series  # noqa: E402
from app.lag_correlation import (  # noqa: E402
    _match_pairs,
    _select_best_lag,
    scan_lag_correlation,
)
from app.main import app  # noqa: E402
from app.trends import _mann_kendall, scan_trends  # noqa: E402


def test_algorithm_spikes_and_shift() -> None:
    n = 40
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(n)]
    # 单点突增
    values[25] = 20.0
    # 水平偏移：连续 5 个点明显抬升
    values[30:35] = [16.0, 16.1, 15.9, 16.2, 15.8]
    ts = [f"2026-09-20T08:{i:02d}:00Z" for i in range(n)]

    result = analyze_series(ts, values, window_size=20, min_window=5, threshold=3.0, merge_gap=2)

    assert result["anomaly_segment_count"] == 2, result["segments"]
    seg1, seg2 = result["segments"]

    # 单点突增区间
    assert seg1["start_index"] == seg1["end_index"] == 25
    assert seg1["score"]["max_abs_zscore"] >= 3.0
    assert seg1["triggers"][0]["direction"] == "上升"

    # 水平偏移区间，5 个点应被合并为一个区间
    assert seg2["start_index"] == 30 and seg2["end_index"] == 34
    assert seg2["point_count"] == 5
    assert "基线" in seg2["explanation"]

    # 逐点注解
    annotated = result["points"]
    assert len(annotated) == n - 5
    assert all("zscore" in p and "baseline_median" in p for p in annotated)
    print("✓ 算法测试通过：单点突变 + 水平偏移识别与合并正确")


def test_constant_baseline() -> None:
    values = [5.0] * 15 + [8.0, 5.0]
    ts = [f"2026-09-20T09:{i:02d}:00Z" for i in range(len(values))]
    result = analyze_series(ts, values, window_size=10, min_window=5, threshold=3.0, merge_gap=1)
    assert result["anomaly_segment_count"] == 1
    trigger = result["segments"][0]["triggers"][0]
    assert trigger["zscore"] == 999.0
    assert "基线恒定" in trigger["reason"]
    print("✓ 恒定基线（MAD=0）退化为强突变哨兵处理正确")


def test_no_anomaly() -> None:
    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.07, 0.93]
    ts = [f"2026-09-20T10:{i:02d}:00Z" for i in range(len(values))]
    result = analyze_series(ts, values, window_size=5, min_window=3, threshold=5.0, merge_gap=1)
    assert result["anomaly_segment_count"] == 0
    assert result["anomaly_point_count"] == 0
    print("✓ 无异常数据正确返回空区间")


def test_rolling_defaults_unchanged() -> None:
    """rolling（默认）结果保持原有行为，新增字段取默认值。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(30)]
    values[10] = 20.0
    ts = [f"2026-09-20T11:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(ts, values, window_size=20, min_window=5, threshold=3.0)
    assert result["baseline_policy"] == "rolling"
    assert result["rebase_after"] is None
    assert result["baseline_rebuild_count"] == 0
    assert all(
        p["baseline_generation"] == 0 and p["baseline_reset"] is False
        for p in result["points"]
    )
    print("✓ rolling 默认模式：结果与新增默认字段正确")


def test_guarded_isolated_spike_excluded() -> None:
    """guarded 模式：孤立尖刺不得进入后续基线，不触发重建。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(30)]
    values[10] = 20.0  # 孤立尖刺
    ts = [f"2026-09-20T12:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0,
        baseline_policy="guarded",
    )
    assert result["baseline_policy"] == "guarded"
    assert result["rebase_after"] == 5  # 默认取 min_window
    assert result["baseline_rebuild_count"] == 0
    assert [(s["start_index"], s["end_index"]) for s in result["segments"]] == [(10, 10)]

    by_index = {p["index"]: p for p in result["points"]}
    # 尖刺被排除：其后正常点的基线中位数仍贴近 10，离散度未被 20 放大
    assert abs(by_index[11]["baseline_median"] - 10.0) < 0.2
    assert by_index[11]["is_anomaly"] is False
    assert by_index[10]["baseline_reset"] is False
    assert all(p["baseline_generation"] == 0 for p in result["points"])

    # 区间基线窗口不得包含被排除的尖刺：成员来自 0..9（5 个种子 + 5..9 正常点），
    # 时间上界停在尖刺前一个点
    seg = result["segments"][0]
    assert seg["baseline"]["window_size"] == 10
    assert seg["baseline"]["window_end_time"] == "2026-09-20T12:09:00Z"
    print("✓ guarded 孤立尖刺：不污染后续中位数/离散度，且不触发重建")


def test_guarded_rebuild_on_consecutive_anomalies() -> None:
    """guarded 模式：连续异常达到 rebase_after 时重建基线、递增代次但不拆区间。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(40)]
    values[20:30] = [16.0] * 10  # 持续水平偏移
    ts = [f"2026-09-20T13:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0, merge_gap=2,
        baseline_policy="guarded",
    )
    by_index = {p["index"]: p for p in result["points"]}

    # 第 5 个连续异常点 i=24 触发重建；该点仍保持异常、区间不拆分
    assert by_index[24]["baseline_reset"] is True
    assert by_index[24]["is_anomaly"] is True
    assert by_index[24]["baseline_generation"] == 0  # 评分用旧代次
    assert result["baseline_rebuild_count"] >= 1
    assert [(s["start_index"], s["end_index"]) for s in result["segments"]][0] == (20, 24)

    # 重建后 i=25..29 对新基线（全 16）恢复正常，代次为 1
    for i in range(25, 30):
        assert by_index[i]["is_anomaly"] is False
        assert by_index[i]["baseline_generation"] == 1
        assert by_index[i]["baseline_reset"] is False

    # 自定义 rebase_after（>= min_window）：首次重建延后到 i=26
    custom = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0,
        baseline_policy="guarded", rebase_after=7,
    )
    assert custom["rebase_after"] == 7
    resets = [p["index"] for p in custom["points"] if p["baseline_reset"]]
    assert resets[0] == 26
    print("✓ guarded 连续异常：达 rebase_after 重建基线并递增代次，触发点仍异常、区间不拆分")


def _gap_series() -> tuple[list[str], list[float]]:
    """两段规则 5 分钟采样（各 14 点），中间断开 10 小时；基线水位 10。"""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 20, 8, 0, 0, tzinfo=timezone.utc)
    timestamps: list[str] = []
    values: list[float] = []
    t = base
    for _ in range(14):
        timestamps.append(t.isoformat())
        values.append(10.0)
        t += timedelta(minutes=5)
    t = base + timedelta(hours=10)  # 大断档（常规间隔的 120 倍）
    for _ in range(14):
        timestamps.append(t.isoformat())
        values.append(10.0)
        t += timedelta(minutes=5)
    return timestamps, values


def test_gap_aware_split_and_warmup() -> None:
    """启用 gap_reset_factor：断档切分、间隔估计、逐段预热与逐点分段编号。"""
    ts, values = _gap_series()
    values[13] = 20.0  # 第一段末点尖刺

    result = analyze_series(
        ts, values, window_size=10, min_window=5, threshold=3.0,
        merge_gap=2, gap_reset_factor=3.0,
    )
    assert result["gap_reset_factor"] == 3.0
    assert result["estimated_interval_seconds"] == 300.0
    assert result["sampling_segment_count"] == 2
    assert len(result["gap_boundaries"]) == 1
    gap = result["gap_boundaries"][0]
    assert gap["before_index"] == 13 and gap["after_index"] == 14
    # 段 0 末点 = base+65min，段 1 首点 = base+10h，差 535min
    assert gap["gap_seconds"] == 10 * 3600 - 65 * 60
    assert gap["gap_ratio"] == gap["gap_seconds"] / 300.0

    # 两段各自预热 5 个点：评估点为 5..13 与 19..27，共 18 个
    evaluated = [p["index"] for p in result["points"]]
    assert evaluated == list(range(5, 14)) + list(range(19, 28))
    assert result["evaluated_point_count"] == 18
    seg_of = {p["index"]: p["sampling_segment_index"] for p in result["points"]}
    assert all(seg_of[i] == 0 for i in range(5, 14))
    assert all(seg_of[i] == 1 for i in range(19, 28))

    # 第一段尖刺被标记，且区间带分段编号
    assert [(s["start_index"], s["sampling_segment_index"]) for s in result["segments"]] == [(13, 0)]
    print("✓ 断档感知：间隔估计、断档边界、逐段预热与逐点分段编号正确")


def test_gap_aware_baseline_restart_and_no_merge_across_gap() -> None:
    """断档后基线从头：新水位不误报；merge_gap 再大，异常区间也不跨断档合并。"""
    from datetime import datetime, timedelta, timezone

    # 场景一：断档后水位迁移到 20，独立预热后不应报异常
    ts, values = _gap_series()
    values[14:] = [20.0] * 14
    gapped = analyze_series(
        ts, values, window_size=10, min_window=5, threshold=3.0, gap_reset_factor=3.0
    )
    plain = analyze_series(
        ts, values, window_size=10, min_window=5, threshold=3.0
    )
    assert gapped["anomaly_segment_count"] == 0
    # 不启用时，断档后早期点仍拿 10 的旧基线评分，会误报一段
    assert plain["anomaly_segment_count"] >= 1

    # 场景二：断档两侧各有异常，merge_gap=10 也不得合并
    ts2, values2 = _gap_series()
    values2[13] = 30.0  # 段 0 末点
    values2[19] = 30.0  # 段 1 首个可评估点（14..18 预热，19 起评分）
    merged = analyze_series(
        ts2, values2, window_size=10, min_window=5, threshold=3.0,
        merge_gap=10, gap_reset_factor=3.0,
    )
    assert [(s["start_index"], s["end_index"]) for s in merged["segments"]] == [(13, 13), (19, 19)]
    assert [s["sampling_segment_index"] for s in merged["segments"]] == [0, 1]
    # 同参数不启用断档时会被桥接成一个跨断档区间
    merged_plain = analyze_series(
        ts2, values2, window_size=10, min_window=5, threshold=3.0, merge_gap=10
    )
    assert [(s["start_index"], s["end_index"]) for s in merged_plain["segments"]] == [(13, 19)]
    print("✓ 断档感知：断档后基线从头避免水位迁移误报，且异常区间不跨断档合并")


def test_gap_aware_guarded_resets_state() -> None:
    """guarded + 断档：第二段连续异常计数与重建代次从头开始。"""
    ts, values = _gap_series()
    # 第一段放一个长水平偏移触发一次重建；第二段同样偏移
    values[10:14] = [30.0] * 4
    values[23:27] = [30.0] * 4

    result = analyze_series(
        ts, values, window_size=10, min_window=3, threshold=3.0,
        merge_gap=2, baseline_policy="guarded", gap_reset_factor=3.0,
    )
    by_index = {p["index"]: p for p in result["points"]}
    # 第一段的重建点 i=12（10,11,12 连续 3 异常，rebase_after 默认=min_window=3）
    assert by_index[12]["baseline_reset"] is True
    assert by_index[12]["baseline_generation"] == 0  # 触发点按旧代次评分
    # 第二段同样在 i=25 触发重建，且代次从 0 重新计数（未受第一段重建影响）
    assert by_index[25]["baseline_reset"] is True
    assert by_index[25]["baseline_generation"] == 0
    assert by_index[26]["baseline_generation"] == 1  # 重建后首个点用新代次
    # 两段各重建一次
    assert result["baseline_rebuild_count"] == 2
    print("✓ 断档感知 guarded：连续异常计数与重建代次在每段起点重置")


def test_gap_aware_defaults_compatible() -> None:
    """未启用 gap_reset_factor 时响应字段取兼容默认值。"""
    ts, values = _gap_series()
    result = analyze_series(
        ts, values, window_size=10, min_window=5, threshold=3.0, merge_gap=2
    )
    assert result["gap_reset_factor"] is None
    assert result["estimated_interval_seconds"] is None
    assert result["gap_boundaries"] == []
    assert result["sampling_segment_count"] == 1
    assert all(p["sampling_segment_index"] == 0 for p in result["points"])
    assert all(s["sampling_segment_index"] == 0 for s in result["segments"])
    print("✓ 断档感知默认关闭：新增字段全部取兼容默认值")


def test_mann_kendall_ties_quantitative() -> None:
    """重复值方差修正 + 连续性校正的量化校验；全同值窗口退化为无趋势。"""
    # 含并列组 [1,1,2,2,3]：S=8，tie_term=2·1·9×2=36，Var=(300−36)/18=14.6667
    s, var_s, z, p = _mann_kendall([1.0, 1.0, 2.0, 2.0, 3.0])
    assert s == 8
    assert abs(var_s - 264.0 / 18.0) < 1e-9
    assert abs(z - 7.0 / var_s**0.5) < 1e-9  # 连续性校正：(S−1)/√Var
    assert abs(z - 1.8278) < 1e-3
    assert abs(p - 0.0676) < 5e-4

    # 无重复值对照 [1,2,3,4,5]：S=10，Var=300/18=16.6667，修正后方差更小
    s2, var_s2, z2, p2 = _mann_kendall([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s2 == 10
    assert abs(var_s2 - 300.0 / 18.0) < 1e-9
    assert var_s < var_s2  # 重复值修正使方差减小
    assert abs(p2 - 0.0275) < 5e-4

    # 全部相同：Var(S)=0，Z=0，p=1，无趋势
    s3, var_s3, z3, p3 = _mann_kendall([7.0] * 6)
    assert s3 == 0 and var_s3 == 0.0 and z3 == 0.0 and p3 == 1.0

    # 全同值窗口经完整扫描判为平稳、不产生区间
    ts = [f"2026-09-20T07:{i:02d}:00Z" for i in range(6)]
    result = scan_trends(ts, [7.0] * 6, window_size=6, step_size=1)
    assert result["windows"][0]["direction"] == "平稳"
    assert result["windows"][0]["significant"] is False
    assert result["trend_segment_count"] == 0
    print("✓ Mann-Kendall 重复值方差修正与连续性校正量化正确，全同值窗口判平稳")


def test_theil_sen_irregular_sampling() -> None:
    """非等距采样：Theil-Sen 按实际时间差还原真实斜率（值/秒）。"""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 20, 6, 0, 0, tzinfo=timezone.utc)
    offsets = [0, 7, 13, 42, 55, 90, 121, 200, 260, 301]  # 非等距秒偏移
    ts = [(base + timedelta(seconds=o)).isoformat() for o in offsets]
    values = [3.0 + 0.02 * o for o in offsets]  # 真实斜率 0.02 /秒

    result = scan_trends(ts, values, window_size=10, step_size=1, min_slope=0.001)
    assert result["window_count"] == 1
    w = result["windows"][0]
    assert abs(w["slope"] - 0.02) < 1e-12  # 按实际时间差，非按点序号
    assert w["significant"] is True and w["direction"] == "上升"
    assert result["segments"][0]["median_slope"] == w["slope"]

    # 同样的值若按等距点序号理解斜率会失真：换一组非等距时间戳斜率随之改变
    offsets2 = [0, 100, 101, 300, 301, 500, 600, 800, 900, 1000]
    ts2 = [(base + timedelta(seconds=o)).isoformat() for o in offsets2]
    values2 = [1.0 + 0.005 * o for o in offsets2]
    result2 = scan_trends(ts2, values2, window_size=10, step_size=1, min_slope=0.0001)
    assert abs(result2["windows"][0]["slope"] - 0.005) < 1e-12
    print("✓ 非等距采样：Theil-Sen 按实际时间差还原斜率（值/秒）")


def test_trend_direction_switch() -> None:
    """方向切换：先升后降的序列拆分为方向相反的两个趋势区间。"""
    n = 40
    values = [float(i) if i < 20 else float(38 - i) for i in range(n)]
    ts = [f"2026-09-20T05:{i // 60:02d}:{i % 60:02d}Z" for i in range(n)]  # 1 秒间隔

    result = scan_trends(ts, values, window_size=10, step_size=5, alpha=0.05, min_slope=0.5)
    assert result["window_count"] == 7
    assert result["trend_segment_count"] == 2

    up, down = result["segments"]
    assert up["direction"] == "上升" and (up["start_index"], up["end_index"]) == (0, 19)
    assert down["direction"] == "下降" and (down["start_index"], down["end_index"]) == (20, 39)
    assert up["window_count"] == 3 and down["window_count"] == 3
    assert abs(up["median_slope"] - 1.0) < 1e-9  # 1 秒间隔下斜率 ±1 /秒
    assert abs(down["median_slope"] + 1.0) < 1e-9
    assert up["min_p_value"] < 1e-3 and down["min_p_value"] < 1e-3
    # 区间成员窗口方向一致，且跨峰窗口 [15..24] 不显著、不属于任何区间
    assert all(w["direction"] == "上升" for w in up["windows"])
    assert all(w["direction"] == "下降" for w in down["windows"])
    middle = result["windows"][3]
    assert (middle["start_index"], middle["end_index"]) == (15, 24)
    assert middle["significant"] is False and middle["direction"] == "平稳"
    print("✓ 方向切换：上升/下降显著窗口各自合并，跨峰窗口判平稳")


def test_trend_merge_adjacent_and_gap() -> None:
    """合并规则：重叠或首尾相接的同向显著窗口合并；留空档的不合并。"""
    values = [float(i) for i in range(25)]
    ts = [f"2026-09-20T04:{i // 60:02d}:{i % 60:02d}Z" for i in range(25)]

    # step == window_size：窗口首尾相接，同向显著窗口合并为一段
    joined = scan_trends(ts, values[:20], window_size=10, step_size=10, min_slope=0.1)
    assert joined["window_count"] == 2
    assert joined["trend_segment_count"] == 1
    seg = joined["segments"][0]
    assert (seg["start_index"], seg["end_index"]) == (0, 19)
    assert seg["window_count"] == 2 and seg["point_count"] == 20

    # step > window_size：窗口间留空档，即使同向显著也不合并
    gapped = scan_trends(ts, values, window_size=10, step_size=15, min_slope=0.1)
    assert gapped["window_count"] == 2
    assert gapped["trend_segment_count"] == 2
    assert [(s["start_index"], s["end_index"]) for s in gapped["segments"]] == [(0, 9), (15, 24)]
    print("✓ 合并规则：重叠/首尾相接合并，空档不合并")


def test_trend_min_slope_filter() -> None:
    """min_slope 过滤：p 值达标但绝对斜率不足时不判趋势。"""
    values = [0.001 * i for i in range(20)]  # 斜率 0.001 /秒（1 秒间隔）
    ts = [f"2026-09-20T03:{i // 60:02d}:{i % 60:02d}Z" for i in range(20)]

    blocked = scan_trends(ts, values, window_size=10, step_size=5, min_slope=0.01)
    assert blocked["significant_window_count"] == 0
    assert blocked["trend_segment_count"] == 0
    assert all(w["p_value"] <= 0.05 for w in blocked["windows"])  # p 值其实达标

    allowed = scan_trends(ts, values, window_size=10, step_size=5, min_slope=0.0001)
    assert allowed["significant_window_count"] == allowed["window_count"]
    assert allowed["trend_segment_count"] == 1
    print("✓ min_slope 过滤：斜率不达标的显著 p 值窗口不判趋势")


def test_lag_correlation_perfect_lag() -> None:
    """正时滞 + 线性响应：driver 提前 600 秒，最佳 lag=600 且 r=1。"""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc)
    dts = [(base + timedelta(seconds=100 * i)).isoformat() for i in range(11)]
    dvs = [float(i) for i in range(11)]
    rts = [(base + timedelta(seconds=600 + 100 * i)).isoformat() for i in range(11)]
    rvs = [2.0 * i + 3.0 for i in range(11)]

    result = scan_lag_correlation(dts, dvs, rts, rvs, [-600.0, 0.0, 300.0, 600.0], 10.0, 3)
    by_lag = {c["lag_seconds"]: c for c in result["candidates"]}
    assert [c["lag_seconds"] for c in result["candidates"]] == [-600.0, 0.0, 300.0, 600.0]
    assert by_lag[600.0]["pair_count"] == 11
    assert by_lag[600.0]["correlation"] == 1.0
    assert by_lag[600.0]["usable"] is True and by_lag[600.0]["reason"] is None
    assert by_lag[-600.0]["pair_count"] == 0
    assert by_lag[-600.0]["usable"] is False and "min_pairs" in by_lag[-600.0]["reason"]
    assert result["best_lag_seconds"] == 600.0
    assert result["driver_point_count"] == 11 and result["response_point_count"] == 11
    print("✓ 时滞关联：完美时滞 + 仿射响应下 best_lag=600、r=1，候选按 lag 升序")


def test_lag_matching_ties_and_no_reuse() -> None:
    """一对一最近邻：等距取更早 response、已匹配 response 不得复用、按 driver 时间贪心。"""
    # 等距（target 恰在两个 response 正中）取更早 response
    assert _match_pairs([10.0], [100.0], [5.0, 15.0], [50.0, 150.0], 0.0, 5.0) == [
        (100.0, 50.0)
    ]
    # 更早的 driver 先占用唯一 response，后续 driver 不得复用
    assert _match_pairs([9.0, 10.0], [90.0, 100.0], [10.0], [10.0], 0.0, 1.0) == [
        (90.0, 10.0)
    ]
    # tie 取更早 response 后，后一个 driver 才能匹配到较晚 response（对比：取较晚会丢第二对）
    assert _match_pairs(
        [10.0, 15.0], [1.0, 2.0], [5.0, 15.0], [10.0, 20.0], 0.0, 5.0
    ) == [(1.0, 10.0), (2.0, 20.0)]
    # 容差为闭区间：距离恰好等于容差仍可配对
    assert _match_pairs([0.0], [1.0], [3.0], [2.0], 0.0, 3.0) == [(1.0, 2.0)]
    assert _match_pairs([0.0], [1.0], [3.0001], [2.0], 0.0, 3.0) == []
    # 正负 lag 平移
    assert _match_pairs([0.0], [1.0], [100.0], [7.0], 100.0, 0.0) == [(1.0, 7.0)]
    assert _match_pairs([100.0], [1.0], [0.0], [7.0], -100.0, 0.0) == [(1.0, 7.0)]
    print("✓ 时滞匹配：等距取更早、不得复用、按时间贪心、容差闭区间与正负 lag 正确")


def test_lag_correlation_unusable_reasons() -> None:
    """候选不可用原因：配对不足、driver/response 侧方差为零；无可用时 best 为 null。"""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 21, 1, 0, 0, tzinfo=timezone.utc)
    ts = [(base + timedelta(seconds=100 * i)).isoformat() for i in range(4)]

    r_const = scan_lag_correlation(ts, [1.0, 2.0, 3.0, 4.0], ts, [5.0] * 4, [0.0], 1.0, 3)
    c = r_const["candidates"][0]
    assert c["usable"] is False and c["correlation"] is None and c["pair_count"] == 4
    assert "response 侧" in c["reason"]

    d_const = scan_lag_correlation(ts, [5.0] * 4, ts, [1.0, 2.0, 3.0, 4.0], [0.0], 1.0, 3)
    assert "driver 侧" in d_const["candidates"][0]["reason"]

    rts_far = [(base + timedelta(seconds=10000 + 100 * i)).isoformat() for i in range(4)]
    few = scan_lag_correlation(ts, [1.0, 2.0, 3.0, 4.0], rts_far, [1.0, 2.0, 3.0, 4.0], [0.0], 1.0, 3)
    assert few["candidates"][0]["usable"] is False
    assert few["candidates"][0]["pair_count"] == 0 and "min_pairs" in few["candidates"][0]["reason"]
    assert few["best_lag_seconds"] is None
    print("✓ 时滞关联：配对不足 / 两侧方差为零原因明确，无可用候选时 best_lag=null")


def test_lag_best_tiebreakers() -> None:
    """best_lag 决胜链：|r|→配对数→|lag|→lag，逐级验证。"""

    def cand(lag: float, r: float, pairs: int) -> dict:
        return {
            "lag_seconds": lag,
            "pair_count": pairs,
            "correlation": r,
            "usable": True,
            "reason": None,
        }

    # 1) |correlation| 更大优先（负相关 -0.9 胜正相关 0.5）
    assert _select_best_lag([cand(0.0, 0.5, 8), cand(10.0, -0.9, 3)]) == 10.0
    # 2) |r| 相同：配对数更多
    assert _select_best_lag([cand(0.0, 1.0, 4), cand(100.0, 1.0, 3)]) == 0.0
    # 3) |r|、配对数相同：|lag| 更小
    assert _select_best_lag([cand(200.0, 1.0, 3), cand(-100.0, 1.0, 3)]) == -100.0
    # 4) 全部相同：lag 更小
    assert _select_best_lag([cand(100.0, 1.0, 3), cand(-100.0, 1.0, 3)]) == -100.0
    # 不可用候选不参与（correlation=None）
    unusable = {"lag_seconds": 0.0, "pair_count": 0, "correlation": None, "usable": False, "reason": "x"}
    assert _select_best_lag([unusable, cand(50.0, 0.3, 2)]) == 50.0
    assert _select_best_lag([unusable]) is None
    print("✓ best_lag 决胜：|correlation|→配对数→|lag|→lag 四级顺序正确")


def test_lag_correlation_negative_correlation() -> None:
    """负相关同样按 |r| 选中：完全反相序列 r=-1。"""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 21, 2, 0, 0, tzinfo=timezone.utc)
    ts = [(base + timedelta(seconds=60 * i)).isoformat() for i in range(6)]
    result = scan_lag_correlation(
        ts, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ts, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [0.0, 60.0], 5.0, 3,
    )
    assert result["candidates"][0]["correlation"] == -1.0
    assert result["best_lag_seconds"] == 0.0  # 0 秒 6 对胜过 60 秒 5 对
    print("✓ 时滞关联：负相关 r=-1 正常计算并参与 |r| 决胜")


def test_api_end_to_end(tmp_db: str = "/tmp/tsad_smoke.db") -> None:
    import os

    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    os.environ["DB_PATH"] = tmp_db

    # 重新初始化数据库（app 模块已导入，直接调用 init_db）
    from app import database

    database.DB_PATH = tmp_db
    database.init_db(tmp_db)

    with TestClient(app) as client:
        _exercise_api(client)


def _exercise_api(client: "TestClient") -> None:
    # 健康检查
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    # 提交检测
    payload = {
        "series_id": "test-series",
        "window_size": 10,
        "min_window": 3,
        "threshold": 3.0,
        "merge_gap": 1,
        "include_points": True,
        "points": [
            {"timestamp": f"2026-09-20T08:{i:02d}:00", "value": 1.0 + (i % 2) * 0.05}
            for i in range(15)
        ]
        + [{"timestamp": "2026-09-20T08:15:00", "value": 9.0}],
    }
    r = client.post("/api/v1/detections", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    detection_id = body["id"]
    assert body["anomaly_segment_count"] == 1
    assert body["segments"][0]["triggers"][0]["timestamp"] == "2026-09-20T08:15:00+00:00"
    assert len(body["points"]) > 0

    # 按 ID 查询
    r = client.get(f"/api/v1/detections/{detection_id}")
    assert r.status_code == 200
    assert r.json()["id"] == detection_id
    assert r.json()["segments"][0]["baseline"]["window_size"] == 10

    # include_points=false 不返回逐点
    r = client.get(f"/api/v1/detections/{detection_id}", params={"include_points": False})
    assert r.json()["points"] is None

    # 列表 + series_id 过滤
    r = client.get("/api/v1/detections", params={"series_id": "test-series"})
    assert r.status_code == 200 and r.json()["total"] == 1

    # 404
    r = client.get("/api/v1/detections/nonexistent")
    assert r.status_code == 404

    # 参数校验：min_window > window_size
    bad = dict(payload)
    bad["min_window"], bad["window_size"] = 20, 10
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # guarded 模式：回显策略与重建统计，逐点注解带代次/重建标记
    guarded_payload = dict(payload)
    guarded_payload["series_id"] = "test-series-guarded"
    guarded_payload["baseline_policy"] = "guarded"
    guarded_payload["rebase_after"] = 3
    r = client.post("/api/v1/detections", json=guarded_payload)
    assert r.status_code == 201, r.text
    gbody = r.json()
    assert gbody["baseline_policy"] == "guarded"
    assert gbody["rebase_after"] == 3
    assert "baseline_rebuild_count" in gbody
    assert all(
        "baseline_generation" in p and "baseline_reset" in p
        for p in gbody["points"]
    )
    guarded_id = gbody["id"]

    # 历史结果取回时策略字段持久化保留
    r = client.get(f"/api/v1/detections/{guarded_id}")
    assert r.status_code == 200
    assert r.json()["baseline_policy"] == "guarded"
    assert r.json()["baseline_rebuild_count"] == gbody["baseline_rebuild_count"]

    # 参数校验：rebase_after < min_window
    bad = dict(payload)
    bad["baseline_policy"] = "guarded"
    bad["min_window"], bad["rebase_after"] = 3, 2
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # 参数校验：非法 baseline_policy
    bad = dict(payload)
    bad["baseline_policy"] = "strict"
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # 断档感知：两段规则采样 + 中间大断档，返回边界 / 间隔 / 逐点分段编号
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 20, 20, 0, 0, tzinfo=timezone.utc)
    gap_points = []
    t = base
    for i in range(12):
        gap_points.append({"timestamp": t.isoformat(), "value": 10.0 + 0.05 * (i % 2)})
        t += timedelta(minutes=5)
    t = base + timedelta(hours=10)
    for i in range(12):
        gap_points.append({"timestamp": t.isoformat(), "value": 20.0 + 0.05 * (i % 2)})
        t += timedelta(minutes=5)
    gap_payload = {
        "series_id": "test-series-gap",
        "window_size": 10,
        "min_window": 3,
        "threshold": 3.0,
        "merge_gap": 1,
        "gap_reset_factor": 3.0,
        "include_points": True,
        "points": gap_points,
    }
    r = client.post("/api/v1/detections", json=gap_payload)
    assert r.status_code == 201, r.text
    gp = r.json()
    assert gp["gap_reset_factor"] == 3.0
    assert gp["estimated_interval_seconds"] == 300.0
    assert gp["sampling_segment_count"] == 2
    assert len(gp["gap_boundaries"]) == 1
    gb = gp["gap_boundaries"][0]
    assert gb["before_index"] == 11 and gb["after_index"] == 12
    # 断档后水位 20 独立预热，不产生误报
    assert gp["anomaly_segment_count"] == 0
    assert all("sampling_segment_index" in p for p in gp["points"])
    assert {p["sampling_segment_index"] for p in gp["points"]} == {0, 1}

    # 历史结果取回时间断档字段持久化保留
    gap_id = gp["id"]
    r = client.get(f"/api/v1/detections/{gap_id}")
    assert r.status_code == 200
    assert r.json()["sampling_segment_count"] == 2
    assert r.json()["gap_boundaries"][0]["gap_seconds"] == gb["gap_seconds"]

    # 参数校验：启用断档但点数 < 3
    bad = dict(gap_payload)
    bad["points"] = gap_points[:2]
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422
    assert "gap_reset_factor" in r.text

    # 参数校验：gap_reset_factor 必须 > 1
    bad = dict(gap_payload)
    bad["gap_reset_factor"] = 1.0
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # 趋势扫描：先升后降序列，返回方向相反的两个区间
    trend_points = []
    for i in range(40):
        v = 10 + 0.25 * i if i < 20 else 10 + 0.25 * (38 - i)
        trend_points.append(
            {"timestamp": f"2026-09-20T22:{i // 60:02d}:{i % 60:02d}Z", "value": v}
        )
    trend_payload = {
        "series_id": "test-series-trend",
        "window_size": 10,
        "step_size": 5,
        "alpha": 0.05,
        "min_slope": 0.1,
        "points": trend_points,
    }
    r = client.post("/api/v1/trends", json=trend_payload)
    assert r.status_code == 200, r.text
    tb = r.json()
    assert tb["series_id"] == "test-series-trend"
    assert tb["window_count"] == 7
    assert tb["significant_window_count"] == 6
    assert tb["trend_segment_count"] == 2
    assert [s["direction"] for s in tb["segments"]] == ["上升", "下降"]
    up_seg, down_seg = tb["segments"]
    assert (up_seg["start_index"], up_seg["end_index"]) == (0, 19)
    assert (down_seg["start_index"], down_seg["end_index"]) == (20, 39)
    assert up_seg["window_count"] == 3 and up_seg["min_p_value"] < 1e-3
    assert abs(up_seg["median_slope"] - 0.25) < 1e-9  # 1 秒间隔，0.25/秒
    assert abs(down_seg["median_slope"] + 0.25) < 1e-9
    assert len(up_seg["windows"]) == 3  # 成员窗口随区间返回
    assert {"s", "var_s", "z", "p_value", "slope", "significant"} <= set(up_seg["windows"][0])
    assert tb["windows"] is None  # 默认不返回全部窗口

    # include_windows=true 返回全部评估窗口（含不显著窗口）
    r = client.post("/api/v1/trends", json={**trend_payload, "include_windows": True})
    assert r.status_code == 200, r.text
    tw = r.json()["windows"]
    assert len(tw) == 7
    assert tw[3]["significant"] is False and tw[3]["direction"] == "平稳"

    # 参数校验：点数少于 window_size
    bad = dict(trend_payload)
    bad["points"] = trend_points[:5]
    r = client.post("/api/v1/trends", json=bad)
    assert r.status_code == 422

    # 参数校验：window_size < 3 / step_size < 1 / alpha 越界 / min_slope 为负
    for field, value in [
        ("window_size", 2),
        ("step_size", 0),
        ("alpha", 0),
        ("alpha", 1),
        ("min_slope", -0.1),
    ]:
        bad = dict(trend_payload)
        bad[field] = value
        r = client.post("/api/v1/trends", json=bad)
        assert r.status_code == 422, (field, value, r.text)

    # 参数校验：重复时间戳
    bad = dict(trend_payload)
    bad["points"] = trend_points[:10] + [trend_points[9]]
    r = client.post("/api/v1/trends", json=bad)
    assert r.status_code == 422
    assert "重复时间戳" in r.text

    # 时滞关联：driver 提前 600 秒线性驱动 response
    lag_driver = [
        {"timestamp": (base + timedelta(seconds=100 * i)).isoformat(), "value": float(i)}
        for i in range(8)
    ]
    lag_response = [
        {"timestamp": (base + timedelta(seconds=600 + 100 * i)).isoformat(), "value": 2.0 * i + 1.0}
        for i in range(8)
    ]
    lag_payload = {
        "driver_points": lag_driver,
        "response_points": lag_response,
        "lag_seconds": [-600, 0, 300, 600, 1200],
        "match_tolerance_seconds": 10,
        "min_pairs": 3,
    }
    r = client.post("/api/v1/lag-correlation", json=lag_payload)
    assert r.status_code == 200, r.text
    lb = r.json()
    assert lb["driver_point_count"] == 8 and lb["response_point_count"] == 8
    assert lb["candidate_lag_count"] == 5 and lb["min_pairs"] == 3
    assert [c["lag_seconds"] for c in lb["candidates"]] == [-600.0, 0.0, 300.0, 600.0, 1200.0]
    by_lag = {c["lag_seconds"]: c for c in lb["candidates"]}
    assert by_lag[600.0]["pair_count"] == 8
    assert by_lag[600.0]["correlation"] == 1.0
    assert by_lag[600.0]["usable"] is True and by_lag[600.0]["reason"] is None
    # lag=300 平移后 driver 后 5 点与 response 前 5 点重合（5 对、线性、可用）；
    # lag=±600 偏差方向不同：-600 无重合，0/1200 各 2 对低于 min_pairs
    assert by_lag[300.0]["pair_count"] == 5 and by_lag[300.0]["usable"] is True
    assert by_lag[-600.0]["usable"] is False
    assert by_lag[-600.0]["correlation"] is None and "min_pairs" in by_lag[-600.0]["reason"]
    assert lb["best_lag_seconds"] == 600.0  # 同为 r=1，8 对胜过 5 对
    assert {"lag_seconds", "pair_count", "correlation", "usable", "reason"} <= set(lb["candidates"][0])

    # 常量 response：配对足够但方差为零，候选不可用且 best 为 null
    const_payload = {
        "driver_points": lag_driver,
        "response_points": [
            {"timestamp": (base + timedelta(seconds=100 * i)).isoformat(), "value": 7.0}
            for i in range(8)
        ],
        "lag_seconds": [0],
        "match_tolerance_seconds": 10,
        "min_pairs": 3,
    }
    r = client.post("/api/v1/lag-correlation", json=const_payload)
    assert r.status_code == 200, r.text
    cb = r.json()
    assert cb["candidates"][0]["usable"] is False
    assert cb["candidates"][0]["correlation"] is None
    assert "方差为零" in cb["candidates"][0]["reason"]
    assert cb["best_lag_seconds"] is None

    # 输入无需预排序；时间戳同时支持 ISO 字符串与 epoch 数字
    unordered = {
        "driver_points": list(reversed(lag_driver)),
        "response_points": [
            {"timestamp": int((base + timedelta(seconds=600 + 100 * i)).timestamp()),
             "value": 2.0 * i + 1.0}
            for i in range(8)
        ],
        "lag_seconds": [600],
        "match_tolerance_seconds": 10,
        "min_pairs": 3,
    }
    r = client.post("/api/v1/lag-correlation", json=unordered)
    assert r.status_code == 200, r.text
    assert r.json()["best_lag_seconds"] == 600.0

    # 参数校验：空序列 / lag 列表为空 / lag 重复或为 NaN / 负容差 / min_pairs<2
    bad = dict(lag_payload)
    bad["driver_points"] = []
    assert client.post("/api/v1/lag-correlation", json=bad).status_code == 422
    bad = dict(lag_payload)
    bad["response_points"] = lag_response[:0]
    assert client.post("/api/v1/lag-correlation", json=bad).status_code == 422
    bad = dict(lag_payload)
    bad["lag_seconds"] = []
    assert client.post("/api/v1/lag-correlation", json=bad).status_code == 422
    bad = dict(lag_payload)
    bad["lag_seconds"] = [0, 0]
    assert client.post("/api/v1/lag-correlation", json=bad).status_code == 422
    # lag 含 NaN / Infinity（非合法 JSON 数值）：直接以原始请求体发送，
    # 服务端解析后由 Pydantic 拒绝（错误响应只回显原始字符串，不二次序列化浮点）
    base_body = json.dumps({k: v for k, v in lag_payload.items() if k != "lag_seconds"})
    for bad_token in ("NaN", "Infinity", "-Infinity"):
        r = client.post(
            "/api/v1/lag-correlation",
            content=base_body[:-1] + f', "lag_seconds": [0, {bad_token}]}}}}',
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422, (bad_token, r.status_code, r.text)
    bad = dict(lag_payload)
    bad["match_tolerance_seconds"] = -1
    assert client.post("/api/v1/lag-correlation", json=bad).status_code == 422
    bad = dict(lag_payload)
    bad["min_pairs"] = 1
    assert client.post("/api/v1/lag-correlation", json=bad).status_code == 422
    # min_pairs 超过上限同样 422
    bad = dict(lag_payload)
    bad["min_pairs"] = 100_001
    assert client.post("/api/v1/lag-correlation", json=bad).status_code == 422
    # 单侧序列内部重复时间戳
    bad = dict(lag_payload)
    bad["driver_points"] = lag_driver + [lag_driver[0]]
    r = client.post("/api/v1/lag-correlation", json=bad)
    assert r.status_code == 422 and "重复时间戳" in r.text

    print("✓ API 端到端测试通过：提交 / 查询 / 列表 / 404 / 参数校验 / guarded 策略 / 趋势扫描 / 时滞关联")


if __name__ == "__main__":
    test_algorithm_spikes_and_shift()
    test_constant_baseline()
    test_no_anomaly()
    test_rolling_defaults_unchanged()
    test_guarded_isolated_spike_excluded()
    test_guarded_rebuild_on_consecutive_anomalies()
    test_gap_aware_split_and_warmup()
    test_gap_aware_baseline_restart_and_no_merge_across_gap()
    test_gap_aware_guarded_resets_state()
    test_gap_aware_defaults_compatible()
    test_mann_kendall_ties_quantitative()
    test_theil_sen_irregular_sampling()
    test_trend_direction_switch()
    test_trend_merge_adjacent_and_gap()
    test_trend_min_slope_filter()
    test_lag_correlation_perfect_lag()
    test_lag_matching_ties_and_no_reuse()
    test_lag_correlation_unusable_reasons()
    test_lag_best_tiebreakers()
    test_lag_correlation_negative_correlation()
    test_api_end_to_end()
    print("\n全部冒烟测试通过 ✅")
