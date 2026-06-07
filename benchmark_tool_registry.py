"""Micro-benchmark for ToolRegistry caching (list_tools + format_for_prompt)."""
import time

from core.tools import ToolRegistry, Tool

REGISTRY_CALLS = 2000


def bench_list_tools(reg: ToolRegistry) -> dict:
    t0 = time.perf_counter()
    for _ in range(REGISTRY_CALLS):
        reg.list_tools()
    elapsed = time.perf_counter() - t0
    return {"list_tools": elapsed, "per_call_us": elapsed / REGISTRY_CALLS * 1e6}


def bench_format_for_prompt(reg: ToolRegistry) -> dict:
    t0 = time.perf_counter()
    for _ in range(REGISTRY_CALLS):
        reg.format_for_prompt()
    elapsed = time.perf_counter() - t0
    return {"format_for_prompt": elapsed, "per_call_us": elapsed / REGISTRY_CALLS * 1e6}


def main():
    reg = ToolRegistry()
    # Warm up to populate cache
    reg.list_tools()
    reg.format_for_prompt()

    runs = 5
    lt_results = []
    fp_results = []
    for _ in range(runs):
        lt_results.append(bench_list_tools(reg))
        fp_results.append(bench_format_for_prompt(reg))

    def best(r, key):
        return min(r[key] for r in r)

    print(f"ToolRegistry cache benchmark — {REGISTRY_CALLS} calls/run, {runs} runs")
    lt_best = best(lt_results, "list_tools")
    fp_best = best(fp_results, "format_for_prompt")
    lt_pu = best(lt_results, "per_call_us")
    fp_pu = best(fp_results, "per_call_us")
    print(f"  list_tools:          best={lt_best*1000:.3f}ms  per_call={lt_pu:.1f}us")
    print(f"  format_for_prompt:   best={fp_best*1000:.3f}ms  per_call={fp_pu:.1f}us")
    total = lt_best + fp_best
    print(f"  combined best:       {total*1000:.3f}ms")


if __name__ == "__main__":
    main()
