"use client";

import React, { useMemo, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

type EquityPoint = { date: string; value: number };
type ChartRow = { date: string; agent: number; buyHold: number };
type PerfMetrics = { returnPct: number; sharpe: number; maxDrawdownPct: number };
type RunStats = { rebalances: number; avgTurnover: number } | null;

const NIFTY_50_SYMBOLS: string[] = [
  "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
  "BPCL","BHARTIARTL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
  "HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","ITC","INDUSINDBK","INFY","JSWSTEEL",
  "KOTAKBANK","LT","M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SUNPHARMA","TCS",
  "TATASTEEL","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","LTIM","SHRIRAMFIN",
];

const clampInt = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, Math.trunc(v)));
const clampFloat = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, v));

const toNum = (x: any): number => {
  const v = parseFloat(String(x));
  return Number.isFinite(v) ? v : NaN;
};

const isoToday = () => {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
};

const mergeSeriesIntersection = (agent: EquityPoint[], bh: EquityPoint[]): ChartRow[] => {
  const a = new Map<string, number>();
  for (const p of agent) a.set(p.date, p.value);

  const rows: ChartRow[] = [];
  for (const p of bh) {
    const av = a.get(p.date);
    if (av == null) continue;
    rows.push({ date: p.date, agent: av, buyHold: p.value });
  }
  rows.sort((x, y) => x.date.localeCompare(y.date));
  return rows;
};

const computeMetrics = (series: number[]): PerfMetrics | null => {
  if (!Array.isArray(series) || series.length < 2) return null;
  const start = series[0];
  const end = series[series.length - 1];
  if (!Number.isFinite(start) || !Number.isFinite(end) || start <= 0) return null;

  const returns: number[] = [];
  for (let i = 1; i < series.length; i += 1) {
    const prev = series[i - 1];
    const curr = series[i];
    if (prev > 0 && Number.isFinite(curr)) returns.push(curr / prev - 1);
  }
  const mean = returns.length ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
  const variance = returns.length
    ? returns.reduce((a, b) => a + (b - mean) * (b - mean), 0) / returns.length
    : 0;
  const std = Math.sqrt(Math.max(variance, 0));
  const sharpe = std > 1e-12 ? (mean / std) * Math.sqrt(252) : 0;

  let peak = series[0];
  let maxDd = 0;
  for (const v of series) {
    peak = Math.max(peak, v);
    if (peak > 0) {
      const dd = (peak - v) / peak;
      if (dd > maxDd) maxDd = dd;
    }
  }

  return {
    returnPct: ((end / start) - 1) * 100,
    sharpe,
    maxDrawdownPct: maxDd * 100,
  };
};

const fmtPct = (v?: number | null) => (Number.isFinite(v as number) ? `${(v as number).toFixed(2)}%` : "—");
const fmtNum = (v?: number | null) => (Number.isFinite(v as number) ? (v as number).toFixed(2) : "—");
const fmtTurn = (v?: number | null) => (Number.isFinite(v as number) ? (v as number).toFixed(4) : "—");
const fmtMoney = (v?: number | null) =>
  Number.isFinite(v as number) ? Math.round(v as number).toLocaleString() : "—";

const ChartingDashboard: React.FC = () => {
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(["TCS"]);
  const [filter, setFilter] = useState<string>("");
  const FIXED_REBALANCE = "weekly";
  const [maxPositions, setMaxPositions] = useState<number>(0); // 0 => full 14
  const [nameRotationFactor, setNameRotationFactor] = useState<number>(0.1);
  const [initialCash, setInitialCash] = useState<number>(100000);
  const [isDark, setIsDark] = useState<boolean>(false);

  const [startDate, setStartDate] = useState<string>("2025-01-01");
  const [endDate, setEndDate] = useState<string>(isoToday());

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("Ready");

  const [chartData, setChartData] = useState<ChartRow[]>([]);
  const [runStats, setRunStats] = useState<RunStats>(null);

  const API_BASE = useMemo(() => {
    return (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/$/, "");
  }, []);

  const filteredUniverse = useMemo(() => {
    const f = filter.trim().toUpperCase();
    if (!f) return NIFTY_50_SYMBOLS;
    return NIFTY_50_SYMBOLS.filter((s) => s.includes(f));
  }, [filter]);

  const canRun = !loading && selectedSymbols.length === 14;
  const effectiveMaxPositions = maxPositions === 0 ? 0 : Math.min(clampInt(maxPositions, 1, 10), selectedSymbols.length);

  const perf = useMemo(() => {
    if (!chartData.length) return { agent: null, bh: null } as { agent: PerfMetrics | null; bh: PerfMetrics | null };
    const agent = computeMetrics(chartData.map((r) => r.agent));
    const bh = computeMetrics(chartData.map((r) => r.buyHold));
    return { agent, bh };
  }, [chartData]);

  const finalEquity = useMemo(() => {
    if (!chartData.length) return { agent: null, bh: null } as { agent: number | null; bh: number | null };
    const last = chartData[chartData.length - 1];
    return { agent: last.agent, bh: last.buyHold };
  }, [chartData]);

  const onSymbolsChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const picked = Array.from(e.target.selectedOptions).map((o) => o.value);
    if (picked.length > 14) {
      setError("Max 14 stocks allowed.");
      return;
    }
    setError(null);
    setSelectedSymbols(picked);
  };

  const removeSymbol = (sym: string) => setSelectedSymbols((prev) => prev.filter((s) => s !== sym));

  const autoFillTo14 = () => {
    setSelectedSymbols((prev) => {
      if (prev.length >= 14) return prev;
      const remaining = NIFTY_50_SYMBOLS.filter((s) => !prev.includes(s));
      return [...prev, ...remaining.slice(0, 14 - prev.length)];
    });
    setError(null);
  };

  const runAgent = async () => {
    setError(null);

    if (!canRun) {
      setError("Select exactly 14 stocks (or use Auto-fill).");
      return;
    }
    if (startDate > endDate) {
      setError("Start date must be <= end date.");
      return;
    }

    setLoading(true);
    setStatus("Running...");

    try {
      const res = await axios.post(
        `${API_BASE}/api/v1/agent/run`,
        {
          symbols: selectedSymbols,
          start_date: startDate,
          end_date: endDate,
          rebalance: FIXED_REBALANCE,
          max_positions: effectiveMaxPositions,
          rotation_factor: clampFloat(Number(nameRotationFactor) || 0, 0, 2),
          initial_cash: Number(initialCash) || 100000,
        },
        { timeout: 120000 }
      );

      const agentSeries: EquityPoint[] =
        res.data?.agent ?? res.data?.agent_equity ?? res.data?.equity_agent ?? [];
      const bhSeries: EquityPoint[] =
        res.data?.buy_and_hold ??
        res.data?.buyHold ??
        res.data?.buy_and_hold_equity ??
        res.data?.equity_benchmark ??
        [];

      if (!Array.isArray(agentSeries) || !agentSeries.length) throw new Error("No agent equity returned.");
      if (!Array.isArray(bhSeries) || !bhSeries.length) throw new Error("No buy-and-hold equity returned.");

      const normAgent: EquityPoint[] = agentSeries
        .map((p: any) => ({
          date: String(p?.date ?? p?.Date ?? ""),
          value: toNum(p?.value ?? p?.PortfolioValue ?? p?.equity ?? p?.Value),
        }))
        .filter((p) => p.date && Number.isFinite(p.value));

      const normBh: EquityPoint[] = bhSeries
        .map((p: any) => ({
          date: String(p?.date ?? p?.Date ?? ""),
          value: toNum(p?.value ?? p?.PortfolioValue ?? p?.equity ?? p?.Value),
        }))
        .filter((p) => p.date && Number.isFinite(p.value));

      if (!normAgent.length) throw new Error("No valid agent equity points.");
      if (!normBh.length) throw new Error("No valid buy-and-hold equity points.");

      const merged = mergeSeriesIntersection(normAgent, normBh);
      if (!merged.length) throw new Error("No overlapping dates between agent and buy&hold.");

      const ep = res.data?.episode ?? {};
      const rebalances = Number(ep?.rebalances ?? 0);
      const avgTurnoverFromApi = Number(ep?.avg_turnover_per_rebalance);
      const turnover = Number(ep?.turnover ?? 0);
      const avgTurnover = Number.isFinite(avgTurnoverFromApi)
        ? avgTurnoverFromApi
        : (rebalances > 0 ? turnover / rebalances : 0);

      setChartData(merged);
      setRunStats({
        rebalances: Number.isFinite(rebalances) ? rebalances : 0,
        avgTurnover: Number.isFinite(avgTurnover) ? avgTurnover : 0,
      });
      setStatus("Done");
    } catch (err: any) {
      console.error(err);
      setChartData([]);
      setRunStats(null);
      setStatus("Failed");
      const detail = err?.response?.data?.detail;
      const httpStatus = err?.response?.status;
      setError(detail || err?.message || `Request failed (HTTP ${httpStatus ?? "n/a"}).`);
    } finally {
      setLoading(false);
    }
  };

  const cardClass = isDark
    ? "rounded-lg border border-slate-700 bg-slate-800 p-3"
    : "rounded-lg border border-gray-200 bg-gray-50 p-3";
  const inputClass = isDark
    ? "w-full px-2 py-2 border border-slate-700 bg-slate-800 text-slate-100 rounded"
    : "w-full px-2 py-2 border border-gray-300 bg-white text-gray-900 rounded";
  const chipClass = isDark
    ? "px-2 py-1 text-xs rounded border border-slate-700 bg-slate-800 hover:bg-slate-700"
    : "px-2 py-1 text-xs rounded border bg-gray-50 hover:bg-gray-100";
  const muted = isDark ? "text-slate-400" : "text-gray-500";
  const label = isDark ? "text-slate-300" : "text-gray-600";
  const title = isDark ? "text-slate-100" : "text-gray-800";
  const gridStroke = isDark ? "#334155" : "#e5e7eb";
  const axisColor = isDark ? "#cbd5e1" : "#475569";

  return (
    <div className={`p-5 shadow-lg rounded-lg transition-colors ${isDark ? "bg-slate-900 text-slate-100" : "bg-white text-gray-900"}`}>
      <div className="flex items-center justify-between gap-3 mb-4">
        <div>
          <div className={`text-xl font-semibold ${title}`}>Trading Agent Dashboard</div>
          <div className={`text-xs ${muted}`}>
            Select exactly 14 stocks (or Auto-fill) • Max positions: 0 (full 14) or 1-10 • Rebalance fixed to weekly
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className={`text-sm ${isDark ? "text-slate-200" : "text-gray-700"}`}>
            <span className="font-semibold">{status}</span>
          </div>
          <button
            type="button"
            onClick={() => setIsDark((v) => !v)}
            className={isDark ? "px-3 py-1.5 rounded border border-slate-600 bg-slate-800 hover:bg-slate-700 text-sm" : "px-3 py-1.5 rounded border border-gray-300 bg-gray-50 hover:bg-gray-100 text-sm"}
          >
            {isDark ? "Light Mode" : "Dark Mode"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <div className={`text-sm font-semibold ${title}`}>Nifty 50</div>
            <div className={`flex items-center gap-2 text-xs ${muted}`}>
              <div>Selected <span className="font-semibold">{selectedSymbols.length}</span>/14</div>
              <button
                type="button"
                onClick={autoFillTo14}
                disabled={loading || selectedSymbols.length >= 14}
                className={`${chipClass} ${loading || selectedSymbols.length >= 14 ? "opacity-60 cursor-not-allowed" : ""}`}
                title="Auto-fill remaining symbols to reach 14"
              >
                Auto-fill to 14
              </button>
            </div>
          </div>

          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Search symbol"
            className={`${inputClass} mb-2 px-3`}
          />

          <select
            multiple
            value={selectedSymbols}
            onChange={onSymbolsChange}
            className={`${inputClass} h-40 px-3`}
          >
            {filteredUniverse.map((sym) => (
              <option key={sym} value={sym}>{sym}</option>
            ))}
          </select>

          <div className="mt-2 flex flex-wrap gap-2">
            {selectedSymbols.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => removeSymbol(s)}
                className={chipClass}
                title="Remove"
              >
                {s} x
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={`block text-xs mb-1 ${label}`}>From</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className={inputClass}
              />
            </div>
            <div>
              <label className={`block text-xs mb-1 ${label}`}>To</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className={inputClass}
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 mt-3">
            <div>
              <label className={`block text-xs mb-1 ${label}`}>Max positions</label>
              <input
                type="number"
                min={0}
                max={10}
                value={maxPositions}
                onChange={(e) => setMaxPositions(clampInt(Number(e.target.value), 0, 10))}
                className={inputClass}
              />
              <div className={`text-[11px] mt-1 ${muted}`}>0 = Full 14</div>
            </div>
            <div>
              <label className={`block text-xs mb-1 ${label}`}>Rotation factor</label>
              <input
                type="number"
                min={0}
                max={2}
                step="0.05"
                value={nameRotationFactor}
                onChange={(e) => setNameRotationFactor(clampFloat(Number(e.target.value), 0, 2))}
                className={inputClass}
              />
            </div>
            <div>
              <label className={`block text-xs mb-1 ${label}`}>Initial cash</label>
              <input
                type="number"
                min={1}
                value={initialCash}
                onChange={(e) => setInitialCash(Number(e.target.value) || 100000)}
                className={inputClass}
              />
            </div>
          </div>

          <button
            onClick={runAgent}
            disabled={!canRun}
            className={`mt-3 w-full px-3 py-2 rounded text-white ${
              canRun ? "bg-blue-600 hover:bg-blue-700" : "bg-blue-600 opacity-60 cursor-not-allowed"
            }`}
          >
            {loading ? "Running..." : "Run Agent"}
          </button>

          {error && (
            <div className={`mt-3 text-sm rounded p-2 border ${isDark ? "text-red-300 bg-red-950 border-red-900" : "text-red-700 bg-red-50 border-red-200"}`}>
              {error}
            </div>
          )}
        </div>
      </div>

      <div style={{ width: "100%", height: 360 }}>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis
                dataKey="date"
                tickFormatter={(d) => String(d).slice(5)}
                tick={{ fill: axisColor, fontSize: 12 }}
                axisLine={{ stroke: gridStroke }}
                tickLine={{ stroke: gridStroke }}
              />
              <YAxis
                type="number"
                tickFormatter={(v) => Math.round(Number(v)).toLocaleString()}
                domain={[(min: number) => min * 0.98, (max: number) => max * 1.02]}
                tick={{ fill: axisColor, fontSize: 12 }}
                axisLine={{ stroke: gridStroke }}
                tickLine={{ stroke: gridStroke }}
              />
              <Tooltip
                formatter={(value: any) => {
                  const n = toNum(value);
                  return Number.isFinite(n) ? Math.round(n).toLocaleString() : String(value);
                }}
                contentStyle={{
                  backgroundColor: isDark ? "#0f172a" : "#ffffff",
                  border: `1px solid ${gridStroke}`,
                  color: isDark ? "#e2e8f0" : "#0f172a",
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="agent" name="Agent" stroke="#2563EB" dot={false} />
              <Line type="monotone" dataKey="buyHold" name="Buy & Hold" stroke="#16A34A" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className={`text-center p-16 ${muted}`}>Run the agent to see Agent vs Buy & Hold.</div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mt-4">
        <div className={cardClass}>
          <div className={`text-xs ${muted}`}>Return</div>
          <div className="text-sm mt-1">Agent: <span className="font-semibold">{fmtPct(perf.agent?.returnPct)}</span></div>
          <div className="text-sm">Buy & Hold: <span className="font-semibold">{fmtPct(perf.bh?.returnPct)}</span></div>
        </div>
        <div className={cardClass}>
          <div className={`text-xs ${muted}`}>Sharpe</div>
          <div className="text-sm mt-1">Agent: <span className="font-semibold">{fmtNum(perf.agent?.sharpe)}</span></div>
          <div className="text-sm">Buy & Hold: <span className="font-semibold">{fmtNum(perf.bh?.sharpe)}</span></div>
        </div>
        <div className={cardClass}>
          <div className={`text-xs ${muted}`}>Max Drawdown</div>
          <div className="text-sm mt-1">Agent: <span className="font-semibold">{fmtPct(perf.agent?.maxDrawdownPct)}</span></div>
          <div className="text-sm">Buy & Hold: <span className="font-semibold">{fmtPct(perf.bh?.maxDrawdownPct)}</span></div>
        </div>
        <div className={cardClass}>
          <div className={`text-xs ${muted}`}>Final Equity</div>
          <div className="text-sm mt-1">Agent: <span className="font-semibold">{fmtMoney(finalEquity.agent)}</span></div>
          <div className="text-sm">Buy & Hold: <span className="font-semibold">{fmtMoney(finalEquity.bh)}</span></div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
        <div className={cardClass}>
          <div className={`text-xs ${muted}`}>Average Turnover (per rebalance)</div>
          <div className="text-sm mt-1"><span className="font-semibold">{fmtTurn(runStats?.avgTurnover)}</span></div>
        </div>
        <div className={cardClass}>
          <div className={`text-xs ${muted}`}>Rebalances</div>
          <div className="text-sm mt-1"><span className="font-semibold">{Number.isFinite(runStats?.rebalances as number) ? runStats?.rebalances : "—"}</span></div>
        </div>
      </div>
    </div>
  );
};

export default ChartingDashboard;
