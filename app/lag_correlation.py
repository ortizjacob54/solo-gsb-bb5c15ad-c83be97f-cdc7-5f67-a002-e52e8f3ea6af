"""双序列时滞关联扫描：候选时滞平移 + 一对一最近邻匹配 + Pearson 相关。

思路：
1. 对每个候选时滞 ``lag``（秒），把 driver 时间整体平移：
   ``t' = t_driver + lag``（lag>0 表示 driver 提前 lag 秒作用于 response）；
2. 按时间顺序（driver 平移后的时间升序）逐个处理 driver 点，在 response
   序列上寻找时间距离最近的点：``|t_response − t'|`` 最小且不超过
   ``match_tolerance_seconds`` 才配对；距离相同（平移点恰在两个 response
   点正中）时取**时间更早**的 response；
3. **一对一匹配**：每个 response 点最多被一个 driver 点使用，配对后即从
   候选池中移除（后续 driver 不可见）。因此匹配顺序会影响结果，统一按
   driver 时间升序做贪心匹配；driver 点找不到容差内的未占用 response
   时直接跳过，不占用任何点；
4. 仅当配对数 ``>= min_pairs`` **且**两侧配对值方差均非零时才计算
   Pearson 相关系数（``min_pairs`` 至少为 2）；否则该候选标记为不可用，
   ``correlation=null`` 并给出原因；
5. ``best_lag`` 在所有可用候选中按：``|correlation|`` 更大 → 配对数更多
   → ``|lag|`` 更小 → ``lag`` 更小 的顺序决出；无可用候选时为 ``null``。

匹配数据结构：response 时间升序后用二分查找最近邻，并用一个路径压缩
并查集维护「未占用索引」（前驱查询用反向坐标复用同一实现），删除为惰性
删除，整体复杂度 ``O(n log m + (n + m)·α(m))``
（n=driver 点数，m=response 点数）。
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from .models import parse_timestamp

# Pearson 相关在少于 2 对时无定义
MIN_PAIRS_LOWER_BOUND = 2


def _find_next(parent: list[int], i: int) -> int:
    """并查集查询：parent[k] 给出 ≥k 的首个未删除逻辑索引。

    数组长度为 size+1，size 为越界哨兵（表示该方向已无可用索引），
    带路径压缩。
    """
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != i:
        nxt = parent[i]
        parent[i] = root
        i = nxt
    return root


def _match_pairs(
    driver_times: list[float],
    driver_values: list[float],
    response_times: list[float],
    response_values: list[float],
    lag_seconds: float,
    tolerance_seconds: float,
) -> list[tuple[float, float]]:
    """单个候选时滞下做一对一最近邻匹配，返回 (driver 值, response 值) 对列表。

    driver 按平移后时间升序处理；response 侧最近邻需在容差内，
    距离相同取更早 response，配对成功的 response 立即移除不得复用。
    """
    m = len(response_times)
    # 后继：next_parent[k] 为 ≥k 的首个未占用索引，哨兵 m
    next_parent = list(range(m + 1))
    # 前驱：在反向坐标 r=m-1-i 上复用同一并查集，哨兵同样为 m
    prev_parent = list(range(m + 1))

    def predecessor(i: int) -> int | None:
        """≤i 的最后一个未占用索引；没有时返回 None。"""
        if i < 0:
            return None
        r = _find_next(prev_parent, m - 1 - i)
        return None if r == m else m - 1 - r

    def successor(i: int) -> int | None:
        """≥i 的第一个未占用索引；没有时返回 None。"""
        j = _find_next(next_parent, i)
        return None if j == m else j

    def erase(j: int) -> None:
        """惰性删除 response 索引 j：后继指向 j+1，前驱（反向）指向 j-1。"""
        next_parent[j] = _find_next(next_parent, j + 1)
        rj = m - 1 - j
        prev_parent[rj] = _find_next(prev_parent, rj + 1)

    pairs: list[tuple[float, float]] = []
    # driver_times 已升序，平移常数不改变先后顺序
    for dt, dv in zip(driver_times, driver_values):
        target = dt + lag_seconds
        pos = bisect.bisect_left(response_times, target)
        left = predecessor(pos - 1)
        right = successor(pos)
        candidates = [j for j in (left, right) if j is not None]
        if not candidates:
            continue
        # 距离最小优先；距离相同（target 正中）取更早 response（索引更小）
        j = min(candidates, key=lambda k: (abs(response_times[k] - target), k))
        if abs(response_times[j] - target) <= tolerance_seconds:
            pairs.append((dv, response_values[j]))
            erase(j)
    return pairs


def _pearson(pairs: list[tuple[float, float]]) -> tuple[float | None, str | None]:
    """计算 Pearson 相关系数，返回 (系数, 不可用原因)。

    配对数不足或任一侧方差为 0（常量序列）时相关无定义：
    返回 (None, 原因)。正常结果截断到 [-1, 1] 以消除浮点误差。
    """
    n = len(pairs)
    if n < MIN_PAIRS_LOWER_BOUND:
        return None, f"配对数 {n} 少于 {MIN_PAIRS_LOWER_BOUND}，无法计算相关"

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0.0:
        return None, "driver 侧配对值方差为零（常量序列），Pearson 相关无定义"
    if syy <= 0.0:
        return None, "response 侧配对值方差为零（常量序列），Pearson 相关无定义"

    sxy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    r = sxy / math.sqrt(sxx * syy)
    return max(-1.0, min(1.0, r)), None


def _select_best_lag(candidates: list[dict[str, Any]]) -> float | None:
    """在可用候选中决出 best_lag。

    顺序：|correlation| 更大 → 配对数更多 → |lag| 更小 → lag 更小；
    没有可用候选时返回 None。
    """
    usable = [c for c in candidates if c["usable"]]
    if not usable:
        return None
    best = max(
        usable,
        key=lambda c: (
            abs(c["correlation"]),
            c["pair_count"],
            -abs(c["lag_seconds"]),
            -c["lag_seconds"],
        ),
    )
    return best["lag_seconds"]


def scan_lag_correlation(
    driver_timestamps: list[str],
    driver_values: list[float],
    response_timestamps: list[str],
    response_values: list[float],
    lag_seconds_list: list[float],
    match_tolerance_seconds: float,
    min_pairs: int,
) -> dict[str, Any]:
    """执行时滞关联扫描，返回可 JSON 序列化的完整结果。

    :param driver_timestamps: 已排序、已去重的 driver 侧 ISO-8601 时间戳
    :param driver_values: 与 driver_timestamps 对齐的数值
    :param response_timestamps: 已排序、已去重的 response 侧 ISO-8601 时间戳
    :param response_values: 与 response_timestamps 对齐的数值
    :param lag_seconds_list: 候选时滞（秒），结果按 lag 升序返回
    :param match_tolerance_seconds: 配对容差（秒，≥0），
        ``|t_response − (t_driver + lag)|`` 不超过该值才允许配对
    :param min_pairs: 计算 Pearson 所需的最少配对数（≥2）
    """
    driver_times = [parse_timestamp(ts).timestamp() for ts in driver_timestamps]
    response_times = [parse_timestamp(ts).timestamp() for ts in response_timestamps]

    candidates: list[dict[str, Any]] = []
    for lag in sorted(lag_seconds_list):
        pairs = _match_pairs(
            driver_times=driver_times,
            driver_values=driver_values,
            response_times=response_times,
            response_values=response_values,
            lag_seconds=lag,
            tolerance_seconds=match_tolerance_seconds,
        )
        entry: dict[str, Any] = {
            "lag_seconds": lag,
            "pair_count": len(pairs),
            "correlation": None,
            "usable": False,
            "reason": None,
        }
        if len(pairs) < min_pairs:
            entry["reason"] = (
                f"配对数 {len(pairs)} 少于 min_pairs={min_pairs}，跳过相关计算"
            )
            candidates.append(entry)
            continue
        correlation, reason = _pearson(pairs)
        entry["correlation"] = correlation
        if correlation is None:
            entry["reason"] = reason
        else:
            entry["usable"] = True
        candidates.append(entry)

    return {
        "driver_point_count": len(driver_values),
        "response_point_count": len(response_values),
        "candidate_lag_count": len(candidates),
        "match_tolerance_seconds": match_tolerance_seconds,
        "min_pairs": min_pairs,
        "candidates": candidates,
        "best_lag_seconds": _select_best_lag(candidates),
    }
