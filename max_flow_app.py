# -*- coding: utf-8 -*-
"""
Оцінка максимальної пропускної здатності мережі з використанням
алгоритмів максимального потоку (Ford-Fulkerson, Edmonds-Karp, Dinic).

Інтерактивний веб-додаток на Streamlit.

Запуск:
    pip install streamlit networkx matplotlib pandas
    streamlit run max_flow_app.py
"""

from __future__ import annotations

import io
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st


# =====================================================================
# 1. СТРУКТУРИ ДАНИХ
# =====================================================================

@dataclass
class FlowEdge:
    """Ребро у мережі потоків з пропускною здатністю та поточним потоком."""
    u: str
    v: str
    capacity: int
    flow: int = 0

    @property
    def residual(self) -> int:
        """Залишкова пропускна здатність."""
        return self.capacity - self.flow


@dataclass
class FlowNetwork:
    """Транспортна мережа: вузли, ребра, джерело, стік."""
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str, int]] = field(default_factory=list)
    source: str = ""
    sink: str = ""

    def to_capacity_dict(self) -> Dict[str, Dict[str, int]]:
        """Перетворює список ребер у словник пропускних здатностей."""
        cap: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for u, v, c in self.edges:
            cap[u][v] += c
        return cap


@dataclass
class FlowResult:
    """Результат роботи алгоритму максимального потоку."""
    max_flow: int
    iterations: int
    elapsed_ms: float
    flow_on_edges: Dict[Tuple[str, str], int]
    min_cut_nodes_s: List[str]
    min_cut_nodes_t: List[str]
    min_cut_edges: List[Tuple[str, str, int]]
    paths: List[Tuple[List[str], int]]


# =====================================================================
# 2. ВЛАСНІ РЕАЛІЗАЦІЇ АЛГОРИТМІВ МАКСИМАЛЬНОГО ПОТОКУ
# =====================================================================

def _build_residual(net: FlowNetwork) -> Dict[str, Dict[str, int]]:
    """Будує початкову залишкову мережу."""
    residual: Dict[str, Dict[str, int]] = {n: {} for n in net.nodes}
    cap = net.to_capacity_dict()
    for u in cap:
        for v in cap[u]:
            residual[u][v] = residual[u].get(v, 0) + cap[u][v]
            if u not in residual[v]:
                residual[v][u] = 0
    for n in net.nodes:
        residual.setdefault(n, {})
    return residual


def _bfs_find_path(
    residual: Dict[str, Dict[str, int]],
    source: str,
    sink: str,
) -> Optional[List[str]]:
    """Пошук збільшуючого шляху в ширину (для Edmonds-Karp)."""
    visited = {source}
    parent: Dict[str, str] = {}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u == sink:
            path = [sink]
            while path[-1] != source:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for v, cap in residual.get(u, {}).items():
            if v not in visited and cap > 0:
                visited.add(v)
                parent[v] = u
                queue.append(v)
    return None


def _dfs_find_path(
    residual: Dict[str, Dict[str, int]],
    source: str,
    sink: str,
) -> Optional[List[str]]:
    """Пошук збільшуючого шляху у глибину (для класичного Ford-Fulkerson)."""
    visited = {source}
    stack: List[Tuple[str, List[str]]] = [(source, [source])]
    while stack:
        u, path = stack.pop()
        if u == sink:
            return path
        for v, cap in residual.get(u, {}).items():
            if v not in visited and cap > 0:
                visited.add(v)
                stack.append((v, path + [v]))
    return None


def _bottleneck(residual: Dict[str, Dict[str, int]], path: List[str]) -> int:
    """Мінімальна залишкова пропускна здатність уздовж шляху."""
    return min(residual[path[i]][path[i + 1]] for i in range(len(path) - 1))


def _augment(residual: Dict[str, Dict[str, int]], path: List[str], delta: int) -> None:
    """Оновлює залишкову мережу після проходження потоку delta."""
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        residual[u][v] -= delta
        residual[v][u] = residual[v].get(u, 0) + delta


def _reachable_from_source(
    residual: Dict[str, Dict[str, int]],
    source: str,
) -> List[str]:
    """Множина вузлів, досяжних із s у залишковій мережі (S-сторона мін-розрізу)."""
    visited = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, cap in residual.get(u, {}).items():
            if v not in visited and cap > 0:
                visited.add(v)
                queue.append(v)
    return sorted(visited)


def _build_result(
    net: FlowNetwork,
    residual: Dict[str, Dict[str, int]],
    max_flow: int,
    iterations: int,
    elapsed_ms: float,
    paths: List[Tuple[List[str], int]],
) -> FlowResult:
    """Збирає результат: розподіл потоку по ребрах, мін-розріз."""
    cap = net.to_capacity_dict()
    flow_on_edges: Dict[Tuple[str, str], int] = {}
    for u in cap:
        for v in cap[u]:
            sent = cap[u][v] - residual[u].get(v, 0)
            if sent > 0:
                flow_on_edges[(u, v)] = sent

    s_side = set(_reachable_from_source(residual, net.source))
    t_side = sorted(set(net.nodes) - s_side)
    cut_edges: List[Tuple[str, str, int]] = []
    for u in s_side:
        for v in cap.get(u, {}):
            if v not in s_side and cap[u][v] > 0:
                cut_edges.append((u, v, cap[u][v]))

    return FlowResult(
        max_flow=max_flow,
        iterations=iterations,
        elapsed_ms=elapsed_ms,
        flow_on_edges=flow_on_edges,
        min_cut_nodes_s=sorted(s_side),
        min_cut_nodes_t=t_side,
        min_cut_edges=cut_edges,
        paths=paths,
    )


def ford_fulkerson(net: FlowNetwork) -> FlowResult:
    """
    Класичний Ford-Fulkerson: пошук шляху у глибину.
    Складність O(E * |f*|), де |f*| — значення максимального потоку.
    """
    residual = _build_residual(net)
    max_flow = 0
    iterations = 0
    paths: List[Tuple[List[str], int]] = []
    t0 = time.perf_counter()
    while True:
        path = _dfs_find_path(residual, net.source, net.sink)
        if path is None:
            break
        delta = _bottleneck(residual, path)
        _augment(residual, path, delta)
        max_flow += delta
        iterations += 1
        if len(paths) < 50:
            paths.append((path, delta))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return _build_result(net, residual, max_flow, iterations, elapsed_ms, paths)


def edmonds_karp(net: FlowNetwork) -> FlowResult:
    """
    Edmonds-Karp: Ford-Fulkerson з вибором найкоротшого шляху (BFS).
    Складність O(V * E^2) — поліноміальна навіть для великих мереж.
    """
    residual = _build_residual(net)
    max_flow = 0
    iterations = 0
    paths: List[Tuple[List[str], int]] = []
    t0 = time.perf_counter()
    while True:
        path = _bfs_find_path(residual, net.source, net.sink)
        if path is None:
            break
        delta = _bottleneck(residual, path)
        _augment(residual, path, delta)
        max_flow += delta
        iterations += 1
        if len(paths) < 50:
            paths.append((path, delta))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return _build_result(net, residual, max_flow, iterations, elapsed_ms, paths)


def _dinic_bfs_levels(
    residual: Dict[str, Dict[str, int]],
    source: str,
    sink: str,
) -> Optional[Dict[str, int]]:
    """Будує пошарову мережу для алгоритму Дініца."""
    level = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, cap in residual.get(u, {}).items():
            if v not in level and cap > 0:
                level[v] = level[u] + 1
                queue.append(v)
    return level if sink in level else None


def _dinic_dfs(
    residual: Dict[str, Dict[str, int]],
    u: str,
    sink: str,
    pushed: int,
    level: Dict[str, int],
    it: Dict[str, int],
    nodes_order: Dict[str, List[str]],
) -> int:
    """DFS Дініца: проштовхування потоку по пошаровій мережі."""
    if u == sink:
        return pushed
    while it[u] < len(nodes_order[u]):
        v = nodes_order[u][it[u]]
        cap = residual[u].get(v, 0)
        if cap > 0 and level.get(v, -1) == level[u] + 1:
            sent = _dinic_dfs(
                residual, v, sink, min(pushed, cap), level, it, nodes_order
            )
            if sent > 0:
                residual[u][v] -= sent
                residual[v][u] = residual[v].get(u, 0) + sent
                return sent
        it[u] += 1
    return 0


def dinic(net: FlowNetwork) -> FlowResult:
    """
    Алгоритм Дініца: пошарова мережа + блокуючий потік.
    Складність O(V^2 * E), на одиничних мережах — O(E * sqrt(V)).
    """
    residual = _build_residual(net)
    max_flow = 0
    iterations = 0
    paths: List[Tuple[List[str], int]] = []
    INF = 10 ** 18
    t0 = time.perf_counter()
    while True:
        level = _dinic_bfs_levels(residual, net.source, net.sink)
        if level is None:
            break
        nodes_order = {u: list(residual[u].keys()) for u in residual}
        it = {u: 0 for u in residual}
        while True:
            pushed = _dinic_dfs(
                residual, net.source, net.sink, INF, level, it, nodes_order
            )
            if pushed == 0:
                break
            max_flow += pushed
            iterations += 1
            if len(paths) < 50:
                paths.append(([net.source, "...", net.sink], pushed))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return _build_result(net, residual, max_flow, iterations, elapsed_ms, paths)


ALGORITHMS = {
    "Ford-Fulkerson (DFS)": ford_fulkerson,
    "Edmonds-Karp (BFS)": edmonds_karp,
    "Dinic (пошарова мережа)": dinic,
}


# =====================================================================
# 3. СТАНДАРТНІ ТЕСТОВІ МЕРЕЖІ
# =====================================================================

def network_corporate() -> FlowNetwork:
    """Корпоративна мережа: дата-центр → користувачі через кілька рівнів."""
    nodes = [
        "DC", "CORE1", "CORE2",
        "AGG1", "AGG2", "AGG3", "AGG4",
        "ACC_FIN", "ACC_IT", "ACC_HR", "ACC_DEV",
        "USERS",
    ]
    edges = [
        ("DC", "CORE1", 1000), ("DC", "CORE2", 1000),
        ("CORE1", "AGG1", 600), ("CORE1", "AGG2", 500),
        ("CORE2", "AGG3", 700), ("CORE2", "AGG4", 400),
        ("AGG1", "AGG2", 200),
        ("AGG3", "AGG4", 200),
        ("AGG1", "ACC_FIN", 300), ("AGG2", "ACC_IT", 350),
        ("AGG3", "ACC_HR", 250), ("AGG4", "ACC_DEV", 400),
        ("AGG2", "ACC_DEV", 200), ("AGG3", "ACC_IT", 150),
        ("ACC_FIN", "USERS", 350), ("ACC_IT", "USERS", 500),
        ("ACC_HR", "USERS", 250), ("ACC_DEV", "USERS", 600),
    ]
    return FlowNetwork(nodes=nodes, edges=edges, source="DC", sink="USERS")


def network_data_center() -> FlowNetwork:
    """Двопланова Clos-топологія дата-центру."""
    nodes = [
        "S", "SP1", "SP2", "SP3",
        "L1", "L2", "L3", "L4",
        "T1", "T2", "T3", "T4", "T",
    ]
    edges = [
        ("S", "SP1", 800), ("S", "SP2", 800), ("S", "SP3", 600),
        ("SP1", "L1", 400), ("SP1", "L2", 400),
        ("SP2", "L2", 400), ("SP2", "L3", 400),
        ("SP3", "L3", 300), ("SP3", "L4", 300),
        ("L1", "T1", 250), ("L1", "T2", 250),
        ("L2", "T2", 300), ("L2", "T3", 300),
        ("L3", "T3", 250), ("L3", "T4", 250),
        ("L4", "T4", 200), ("L4", "T1", 200),
        ("T1", "T", 400), ("T2", "T", 500),
        ("T3", "T", 400), ("T4", "T", 350),
    ]
    return FlowNetwork(nodes=nodes, edges=edges, source="S", sink="T")


def network_classic() -> FlowNetwork:
    """Класичний приклад з підручника Cormen et al."""
    nodes = ["s", "v1", "v2", "v3", "v4", "t"]
    edges = [
        ("s", "v1", 16), ("s", "v2", 13),
        ("v1", "v3", 12),
        ("v2", "v1", 4), ("v2", "v4", 14),
        ("v3", "v2", 9), ("v3", "t", 20),
        ("v4", "v3", 7), ("v4", "t", 4),
    ]
    return FlowNetwork(nodes=nodes, edges=edges, source="s", sink="t")


PRESETS = {
    "Корпоративна мережа (12 вузлів)": network_corporate,
    "Дата-центр Clos (13 вузлів)": network_data_center,
    "Класичний приклад Cormen": network_classic,
}


# =====================================================================
# 4. ВІЗУАЛІЗАЦІЯ
# =====================================================================

def draw_network(net: FlowNetwork, result: Optional[FlowResult] = None) -> bytes:
    """Малює мережу з підписами потоків та підсвічує мін-розріз."""
    G = nx.DiGraph()
    for n in net.nodes:
        G.add_node(n)
    for u, v, c in net.edges:
        G.add_edge(u, v, capacity=c)

    pos = nx.spring_layout(G, seed=7, k=1.2)
    fig, ax = plt.subplots(figsize=(12, 8))

    if result:
        s_side = set(result.min_cut_nodes_s)
        node_colors = ["#ffb3b3" if n in s_side else "#b3d9ff" for n in G.nodes]
    else:
        node_colors = ["#dddddd" for _ in G.nodes]
    node_colors_final = []
    for n, c in zip(G.nodes, node_colors):
        if n == net.source:
            node_colors_final.append("#90ee90")
        elif n == net.sink:
            node_colors_final.append("#ffd700")
        else:
            node_colors_final.append(c)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors_final,
                           node_size=1500, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    cut_set = set()
    if result:
        cut_set = {(u, v) for u, v, _ in result.min_cut_edges}

    for (u, v) in G.edges:
        cap = G[u][v]["capacity"]
        flow = result.flow_on_edges.get((u, v), 0) if result else 0
        is_cut = (u, v) in cut_set
        color = "red" if is_cut else ("#1f78b4" if flow > 0 else "#999999")
        width = 3.0 if is_cut else (1.5 + 2.0 * (flow / cap if cap else 0))
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=color, width=width,
            arrows=True, arrowsize=18, ax=ax,
        )

    edge_labels = {}
    for u, v in G.edges:
        cap = G[u][v]["capacity"]
        flow = result.flow_on_edges.get((u, v), 0) if result else 0
        edge_labels[(u, v)] = f"{flow}/{cap}" if result else f"{cap}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax)

    title = f"Мережа: {len(G.nodes)} вузлів, {len(G.edges)} ребер"
    if result:
        title += f"  |  Макс. потік = {result.max_flow}"
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =====================================================================
# 5. ПОРІВНЯЛЬНИЙ АНАЛІЗ
# =====================================================================

def compare_algorithms(net: FlowNetwork) -> pd.DataFrame:
    """Порівнює 3 алгоритми за часом, ітераціями та значенням потоку."""
    rows = []
    for name, fn in ALGORITHMS.items():
        r = fn(net)
        rows.append({
            "Алгоритм": name,
            "Макс. потік": r.max_flow,
            "Ітерацій": r.iterations,
            "Час, мс": round(r.elapsed_ms, 3),
        })
    return pd.DataFrame(rows)


def bottleneck_analysis(net: FlowNetwork, result: FlowResult) -> pd.DataFrame:
    """Аналіз завантаження ребер (виявлення вузьких місць)."""
    cap = net.to_capacity_dict()
    rows = []
    for u in cap:
        for v in cap[u]:
            c = cap[u][v]
            f = result.flow_on_edges.get((u, v), 0)
            util = (f / c * 100) if c else 0
            rows.append({
                "Ребро": f"{u} → {v}",
                "Пропускна здатність": c,
                "Потік": f,
                "Завантаження, %": round(util, 1),
                "Вузьке місце": "ТАК" if util >= 99.5 else "ні",
            })
    df = pd.DataFrame(rows).sort_values("Завантаження, %", ascending=False)
    return df.reset_index(drop=True)


# =====================================================================
# 6. STREAMLIT UI
# =====================================================================

def render_ui() -> None:
    """Точка входу веб-додатку."""
    st.set_page_config(
        page_title="Аналіз пропускної здатності мережі",
        layout="wide",
        page_icon=":satellite:",
    )

    st.title("Оцінка максимальної пропускної здатності мережі")
    st.caption(
        "Курсова робота. Реалізація алгоритмів Ford-Fulkerson, Edmonds-Karp та Дініца."
    )

    with st.sidebar:
        st.header("Параметри моделювання")
        preset = st.selectbox("Тестова мережа", list(PRESETS.keys()))
        algo_name = st.selectbox("Алгоритм", list(ALGORITHMS.keys()))
        st.divider()
        st.subheader("Редагування ребер")
        net = PRESETS[preset]()
        edges_df = pd.DataFrame(net.edges, columns=["from", "to", "capacity"])
        edited = st.data_editor(
            edges_df, num_rows="dynamic", use_container_width=True,
            key=f"editor_{preset}",
        )
        net.edges = [(r["from"], r["to"], int(r["capacity"]))
                     for _, r in edited.iterrows()
                     if r["from"] and r["to"] and r["capacity"]]
        nodes_set = set()
        for u, v, _ in net.edges:
            nodes_set.add(u)
            nodes_set.add(v)
        net.nodes = sorted(nodes_set)
        net.source = st.selectbox("Джерело (s)", net.nodes,
                                  index=net.nodes.index(net.source)
                                  if net.source in net.nodes else 0)
        net.sink = st.selectbox("Стік (t)", net.nodes,
                                index=net.nodes.index(net.sink)
                                if net.sink in net.nodes else len(net.nodes) - 1)
        run = st.button("Розрахувати", type="primary", use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Візуалізація", "Збільшуючі шляхи", "Порівняння алгоритмів",
        "Аналіз вузьких місць",
    ])

    if not run:
        with tab1:
            st.info("Налаштуйте мережу зліва та натисніть «Розрахувати».")
            st.image(draw_network(net), use_container_width=True)
        return

    try:
        algo = ALGORITHMS[algo_name]
        result = algo(net)
    except Exception as exc:
        st.error(f"Помилка під час обчислень: {exc}")
        return

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Макс. потік", f"{result.max_flow}")
        c2.metric("Ітерацій", f"{result.iterations}")
        c3.metric("Час, мс", f"{result.elapsed_ms:.2f}")
        c4.metric("Ребер у розрізі", f"{len(result.min_cut_edges)}")
        st.image(draw_network(net, result), use_container_width=True)
        st.subheader("Мінімальний розріз (s-t cut)")
        st.write(f"**S-сторона:** {', '.join(result.min_cut_nodes_s)}")
        st.write(f"**T-сторона:** {', '.join(result.min_cut_nodes_t)}")
        cut_df = pd.DataFrame(result.min_cut_edges,
                              columns=["from", "to", "capacity"])
        st.dataframe(cut_df, use_container_width=True)

    with tab2:
        st.subheader("Знайдені збільшуючі шляхи")
        if not result.paths:
            st.info("Шляхи не зафіксовано (для Дініца виводиться зведено).")
        for i, (p, d) in enumerate(result.paths, 1):
            st.write(f"{i}. `{' → '.join(p)}` — потік **{d}**")

    with tab3:
        st.subheader("Порівняння алгоритмів на цій мережі")
        df_cmp = compare_algorithms(net)
        st.dataframe(df_cmp, use_container_width=True)
        st.bar_chart(df_cmp.set_index("Алгоритм")["Час, мс"])

    with tab4:
        st.subheader("Завантаження ребер")
        df_b = bottleneck_analysis(net, result)
        st.dataframe(df_b, use_container_width=True)
        n_bottle = (df_b["Вузьке місце"] == "ТАК").sum()
        st.warning(f"Виявлено вузьких місць: **{n_bottle}**")


# =====================================================================
# 7. КОНСОЛЬНИЙ РЕЖИМ (без Streamlit)
# =====================================================================

def run_console_demo() -> None:
    """Демо в консолі — використовується для перевірки без UI."""
    print("=" * 70)
    print("ОЦІНКА МАКСИМАЛЬНОЇ ПРОПУСКНОЇ ЗДАТНОСТІ МЕРЕЖІ")
    print("=" * 70)
    for preset_name, builder in PRESETS.items():
        net = builder()
        print(f"\n>>> Мережа: {preset_name}")
        print(f"    Вузлів: {len(net.nodes)}, ребер: {len(net.edges)}")
        print(f"    Джерело: {net.source}, стік: {net.sink}")
        for algo_name, fn in ALGORITHMS.items():
            r = fn(net)
            print(f"    {algo_name:32s}  потік={r.max_flow:5d}  "
                  f"ітерацій={r.iterations:4d}  час={r.elapsed_ms:7.3f} мс")
        net = builder()
        ek = edmonds_karp(net)
        print(f"    Мін-розріз: {len(ek.min_cut_edges)} ребер, "
              f"сума пропускних здатностей = "
              f"{sum(c for _, _, c in ek.min_cut_edges)}")


def main() -> None:
    try:
        import streamlit.runtime.scriptrunner as _sr  # noqa: F401
        is_streamlit = _sr.get_script_run_ctx() is not None
    except Exception:
        is_streamlit = False

    if is_streamlit:
        render_ui()
    else:
        try:
            run_console_demo()
        except Exception as exc:
            print(f"[ПОМИЛКА] {exc}")
            raise


if __name__ == "__main__":
    main()
else:
    try:
        import streamlit.runtime.scriptrunner as _sr
        if _sr.get_script_run_ctx() is not None:
            render_ui()
    except Exception:
        pass
