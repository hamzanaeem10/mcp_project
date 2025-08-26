from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict, List, Optional, Tuple,Literal, Sequence
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Tuple
import ast
import math
from fractions import Fraction
import json
import re
import os
from functools import lru_cache
load_dotenv()

mcp  = FastMCP("demo-server")

try:
    import sympy as sp  # optional
except Exception:
    sp = None


doc_texts = {}
json_cache = {}
DOCS_FOLDER = "docs"


_ALLOWED_FUNCS = {
    # arithmetic
    "abs": abs, "round": round,
    # math module (whitelist)
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "gcd": math.gcd, "factorial": math.factorial,
    # fractions
    "Fraction": Fraction,
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
    ast.Name, ast.Tuple
)
    
@mcp.tool(name="brave_search")
def search_web(query: str, max_results: int = 5) -> list:
    """
    Perform a web search using Tavily API.
    
    Args:
        query: Search query string
        max_results: Number of results to return (default: 5)
    
    Returns:
        List of dicts with title, url, and content
    """
    import os
    import requests

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{"error": "Missing TAVILY_API_KEY in environment"}]

    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"query": query, "num_results": max_results}
        
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Tavily returns results under 'results'
        return [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content"),
            }
            for item in data.get("results", [])
        ]
    except Exception as e:
        return [{"error": str(e)}]
    
    
@mcp.resource("docs://all")
def documents():
    """
    Expose all files in the docs/ folder as a resource.
    """
    if not os.path.exists(DOCS_FOLDER):
        return {
            "mimeType": "text/plain",
            "text": f"Docs folder not found: {DOCS_FOLDER}"
        }

    files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf")]
    if not files:
        return {
            "mimeType": "text/plain",
            "text": "No PDF files found in docs/"
        }

    return {
        "uri": "docs://all",
        "mimeType": "text/plain",
        "text": "Available documents:\n" + "\n".join(files)
    }

# ---------------------------
# Tool: load and read a PDF
# ---------------------------
@mcp.tool()
def read_document(filename: str) -> str:
    """
    Read and return the text of a PDF file from docs/.
    """
    file_path = os.path.join(DOCS_FOLDER, filename)
    if not os.path.exists(file_path):
        return f"Error: {filename} not found in {DOCS_FOLDER}"

    if filename in doc_texts:
        return doc_texts[filename]

    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        doc_texts[filename] = text
        return text[:2000] + "... [truncated]" if len(text) > 2000 else text
    except Exception as e:
        return f"Error reading {filename}: {str(e)}"
    
# ---------------------------
# Resource: list JSON files
# ---------------------------
@mcp.resource("json://all")
def list_json_files():
    """
    Expose all JSON files in the docs/ folder as a resource.
    """
    files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".json")]
    if not files:
        return {
            "mimeType": "text/plain",
            "text": "No JSON files found in docs/"
        }

    return {
        "uri": "json://all",
        "mimeType": "text/plain",
        "text": "Available JSON documents:\n" + "\n".join(files)
    }

# ---------------------------
# Tool: read JSON content
# ---------------------------
@mcp.tool()
def read_json(filename: str) -> str:
    """
    Read and return the structured data of a JSON file from docs/.
    """
    file_path = os.path.join(DOCS_FOLDER, filename)
    if not os.path.exists(file_path):
        return f"Error: {filename} not found in {DOCS_FOLDER}"

    if filename in json_cache:
        return json.dumps(json_cache[filename], indent=2)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        json_cache[filename] = data
        return json.dumps(data, indent=2)[:2000] + "... [truncated]" if len(json.dumps(data)) > 2000 else json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading {filename}: {str(e)}"
# ---------------------------

def _safe_eval_linear_expr(expr: str, x_value: float | None = None) -> float:
    """
    Evaluate an arithmetic expression that may contain the variable `x`.
    Allowed: numbers, x, + - * / and parentheses. No functions, no power.
    If x_value is None, then 'x' must not appear in the expression.
    """
    tree = ast.parse(expr, mode="eval")

    def _check(node: ast.AST):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id != "x":
            raise ValueError(f"Unknown name: {node.id}")
        for child in ast.iter_child_nodes(node):
            _check(child)
    _check(tree)

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):  # py3.8+
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants allowed.")
        if isinstance(node, ast.Name):
            if node.id == "x":
                if x_value is None:
                    raise ValueError("Expression contains 'x' but no x_value provided.")
                return float(x_value)
        if isinstance(node, ast.UnaryOp):
            v = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):  return +v
            if isinstance(node.op, ast.USub):  return -v
            raise ValueError("Unary op not allowed")
        if isinstance(node, ast.BinOp):
            l = _eval(node.left); r = _eval(node.right)
            if isinstance(node.op, ast.Add):   return l + r
            if isinstance(node.op, ast.Sub):   return l - r
            if isinstance(node.op, ast.Mult):  return l * r
            if isinstance(node.op, ast.Div):   return l / r
            raise ValueError("Unsupported operator")
        raise ValueError("Node not allowed")

    return _eval(tree)



def _coeff_const(expr: str) -> Tuple[float, float]:
    """
    For linear expr f(x) = a*x + b, compute (a, b) by probing:
    a = f(1) - f(0); b = f(0).
    Verify linearity with f(2) check.
    """
    # Normalize common implicit “2x” patterns to keep AST simple (optional but nice):
    # (We actually don't need to because AST handles 2*x if present; implicit 2x remains 'Name' + 'Num' not parsed.
    # So we rely on allowed grammar and expect callers to pass valid pythonic expr like 2*x.)
    # To be user-friendly, convert '2x' -> '2*x' and '-x' -> '-1*x' quickly:
    import re
    s = expr
    s = re.sub(r'(?<![\w.])(\d+(\.\d+)?)[ ]*x', r'\1*x', s)   # 2x -> 2*x
    s = re.sub(r'(?<![\w.])([+\-])x', r'\1 1*x', s)           # +x/-x -> +1*x/-1*x
    s = re.sub(r'(?<![\w.])x', r'1*x', s)                     # leading x -> 1*x

    f0 = _safe_eval_linear_expr(s, 0.0)
    f1 = _safe_eval_linear_expr(s, 1.0)
    f2 = _safe_eval_linear_expr(s, 2.0)
    a = f1 - f0
    # linearity check: f(2) - f(0) ≈ 2 * (f(1) - f(0))
    if abs((f2 - f0) - 2*a) > 1e-9:
        raise ValueError("Expression is not linear in x.")
    b = f0
    return a, b

# ---------- GSM8k ----------

@mcp.tool()
async def calc(expr: str, x_value: float | None = None) -> dict:
    """
    Evaluate a numeric or linear expression safely.
    Examples:
      - "2*(3+4)"
      - "10/5 + 3"
      - "3*x + 5" with x_value=2
    Returns {"expr": <expr>, "result": <number>}
    """
    expr = expr.replace("^", "**")  # normalize ^

    try:
        res = _safe_eval_linear_expr(expr, x_value)
    except Exception as e:
        return {"expr": expr, "error": str(e)}

    return {"expr": expr, "result": float(res)}

@mcp.tool()
async def nCr(n: int, r: int) -> dict:
    """Compute combinations C(n, r). Returns {"n":n,"r":r,"value":int}"""
    if n < 0 or r < 0 or r > n:
        return {"n": n, "r": r, "value": 0}
    # use math.comb if available
    value = math.comb(n, r)
    return {"n": n, "r": r, "value": value}

@mcp.tool()
async def percent_of(percent: float, base: float) -> dict:
    """Return (percent% of base). Example: percent_of(25, 240) -> 60"""
    return {"percent": percent, "base": base, "value": percent * base / 100.0}

@mcp.tool()
async def average(values: Sequence[float]) -> dict:
    """Arithmetic mean of a list of numbers. Returns {"mean": float, "n": int}"""
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "n": 0}
    return {"mean": sum(vals) / len(vals), "n": len(vals)}

def _parse_num(val):
    """
    Convert input into a float if possible.
    Handles:
      - int/float already
      - strings with units ("30 km", "75 km/h")
      - arithmetic expressions ("60*0.8")
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().lower()
        # strip common units
        for token in ["km", "km/h", "kph", "h", "hr", "hours", "hour",
                      "minutes", "minute", "min", "s", "sec", "seconds"]:
            s = s.replace(token, "")
        try:
            # try plain float
            return float(s)
        except Exception:
            # try safe eval of simple arithmetic
            try:
                return float(eval(s, {"__builtins__": {}}, {}))
            except Exception as e:
                raise ValueError(f"Cannot parse value '{val}': {e}")
    return float(val)

@mcp.tool()
async def d_s_t(distance: float | None = None,
                speed: float | None = None,
                time: float | None = None,
                units: str | None = None) -> dict:
    """
    Compute distance, speed, or time given exactly two of them.

    ⚠️ Pass only numbers or simple numeric expressions.
    Examples:
      {"distance": 30, "speed": 48}
      {"distance": "30 km", "speed": "60*0.8"}
    """
    D, S, T = _parse_num(distance), _parse_num(speed), _parse_num(time)

    given = sum(x is not None for x in (D, S, T))
    if given != 2:
        return {
            "error": "Provide exactly two of distance/speed/time.",
            "do_not_retry": True,
            "hint": "This tool is for a single uniform segment only."
        }

    # normal compute
    if D is None:
        D = S * T
    elif S is None:
        S = D / T
    elif T is None:
        T = D / S

    return {"distance": D, "speed": S, "time": T, "units": units}
@mcp.tool()
async def ratio_split(total: float, a: float, b: float) -> dict:
    """
    Split 'total' in ratio a:b. Returns {"part_a":..., "part_b":...}.
    Useful for "Alice and Bob share $T in ratio A:B".
    """
    s = a + b
    if s == 0:
        return {"error": "Ratio sum is zero"}
    return {"part_a": total * a / s, "part_b": total * b / s}

@mcp.tool()
async def solve_linear(equation: str, var: str = "x") -> dict:
    """
    Solve a single linear equation like:
      - "2x + 3 = 11"
      - "3*(x - 2) = 7"
      - "(x/3) + 4 = 6"
    Assumes linear in `var` (default 'x'). Returns:
      {"var": var, "solution": number}
    or {"error": "..."} / {"infinite_solutions": true}
    """
    # We only support variable named `var`; internally we rename to 'x'
    try:
        eq = equation.replace(" ", "")
        if "=" not in eq:
            return {"error": "Equation must contain '='."}
        left_raw, right_raw = eq.split("=", 1)
        # rename target variable to 'x'
        if var != "x":
            left = left_raw.replace(var, "x")
            right = right_raw.replace(var, "x")
        else:
            left, right = left_raw, right_raw

        aL, bL = _coeff_const(left)
        aR, bR = _coeff_const(right)

        a = aL - aR
        b = bL - bR
        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return {"var": var, "infinite_solutions": True}
            else:
                return {"var": var, "error": "No solution"}
        x = -b / a
        # Return nice float; callers can format with your format_answer tool
        return {"var": var, "solution": float(x)}
    except ZeroDivisionError:
        return {"var": var, "error": "Division by zero in equation."}
    except ValueError as e:
        return {"var": var, "error": str(e)}
    except Exception as e:
        return {"var": var, "error": f"Failed to solve: {e}"}

@mcp.tool()
async def format_answer(value: float | int | str,
                        style: Literal["int","float","money","percent","fraction"] = "float",
                        decimals: int = 2) -> dict:
    """
    Format a final numeric answer consistently.
    - int: round to nearest int
    - float: round to `decimals`
    - money: prefix $ and 2 decimals
    - percent: append %
    - fraction: best rational approximation (den<=1000)
    """
    if style == "int":
        return {"answer": str(int(round(float(value))))}
    if style == "money":
        return {"answer": f"${float(value):.2f}"}
    if style == "percent":
        return {"answer": f"{float(value):.{decimals}f}%"}
    if style == "fraction":
        try:
            frac = Fraction(float(value)).limit_denominator(1000)
            return {"answer": f"{frac.numerator}/{frac.denominator}"}
        except Exception:
            return {"answer": str(value)}
    # float default
    return {"answer": f"{float(value):.{decimals}f}"}

_SECONDS_PER_DAY = 24 * 60 * 60

def _parse_hms_time(s: str) -> int:
    """
    Parse wall-clock 'HH:MM' or 'HH:MM:SS' -> total seconds since 00:00:00.
    Hours must be 0..23, minutes 0..59, seconds 0..59.
    """
    s = s.strip()
    parts = s.split(":")
    if not (2 <= len(parts) <= 3):
        raise ValueError("time must be 'HH:MM' or 'HH:MM:SS'")
    h = int(parts[0]); m = int(parts[1]); sec = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59):
        raise ValueError("time components out of range")
    return h * 3600 + m * 60 + sec

_DURATION_TOKEN = re.compile(r"(?P<sign>[+-])?\s*(?:(?P<h>\d+)\s*h)?\s*(?:(?P<m>\d+)\s*m)?\s*(?:(?P<s>\d+)\s*s)?\s*$")

def _parse_duration(delta: Optional[str], hours: int, minutes: int, seconds: int) -> int:
    """
    Parse a duration into total seconds.
    Accepted forms:
      - "H:MM" or "H:MM:SS" (optionally with leading + / -)
      - tokens like "+2h 15m", "-45m 10s", "90s", "1h"
      - or separate integers hours/minutes/seconds (signed allowed)
    """
    if delta and delta.strip():
        s = delta.strip()
        # Accept +HH:MM[:SS] or -HH:MM[:SS]
        if ":" in s:
            sign = 1
            if s[0] in "+-":
                sign = -1 if s[0] == "-" else 1
                s = s[1:]
            parts = s.split(":")
            if not (2 <= len(parts) <= 3):
                raise ValueError("duration must be 'H:MM' or 'H:MM:SS' when using colon form")
            h = int(parts[0]); m = int(parts[1]); sec = int(parts[2]) if len(parts) == 3 else 0
            if m < 0 or m > 59 or sec < 0 or sec > 59:
                raise ValueError("duration minutes/seconds out of range")
            return sign * (h * 3600 + m * 60 + sec)

        # Accept token form: [+/-][#h] [#m] [#s]
        m = _DURATION_TOKEN.match(s)
        if m:
            sign = -1 if (m.group("sign") == "-") else 1
            h = int(m.group("h") or 0)
            mi = int(m.group("m") or 0)
            sec = int(m.group("s") or 0)
            return sign * (h * 3600 + mi * 60 + sec)

        raise ValueError("Unrecognized duration format")
    else:
        # Use numeric fields (may be negative)
        return hours * 3600 + minutes * 60 + seconds

def _format_hms(total_seconds: int, wrap_24h: bool) -> Tuple[str, int]:
    """
    Format total seconds to ('HH:MM:SS', days_offset).
    If wrap_24h: HH is modulo 24 and days_offset counts rolled days (can be negative).
    If not wrap_24h: HH can exceed 24; days_offset = 0.
    """
    if wrap_24h:
        # Python's modulo keeps sign; we want floor div for negative support
        days_offset = total_seconds // _SECONDS_PER_DAY
        sec_in_day = total_seconds % _SECONDS_PER_DAY
        if sec_in_day < 0:
            sec_in_day += _SECONDS_PER_DAY
            days_offset -= 1
        h = sec_in_day // 3600
        rem = sec_in_day % 3600
        m = rem // 60
        s = rem % 60
        return f"{h:02d}:{m:02d}:{s:02d}", days_offset
    else:
        neg = total_seconds < 0
        sec = abs(total_seconds)
        h = sec // 3600
        rem = sec % 3600
        m = rem // 60
        s = rem % 60
        prefix = "-" if neg else ""
        return f"{prefix}{h:02d}:{m:02d}:{s:02d}", 0

@mcp.tool()
async def time_math(
    op: Literal["add", "sub"],
    time: str,
    delta: Optional[str] = None,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    wrap_24h: bool = True
) -> dict:
    """
    Add or subtract a duration to/from a wall-clock time.

    Parameters
    ----------
    op : "add" | "sub"
        Operation to perform.
    time : "HH:MM" | "HH:MM:SS"
        Base time (0<=HH<=23).
    delta : str, optional
        Duration string, examples:
          "1:30"     -> 1h 30m
          "-0:45"    -> -45m
          "02:15:10" -> 2h 15m 10s
          "+2h 5m"   -> 2h 5m
          "-45m 10s" -> -45m 10s
    hours, minutes, seconds : int
        Alternative numeric duration; used only if `delta` not provided.
        Can be negative.
    wrap_24h : bool (default True)
        If True: wrap around 24h and return 'days_offset' for overflow.
        If False: return possibly >24h or negative HH with no wrapping.

    Returns
    -------
    {
      "input_time": "HH:MM:SS",
      "delta_seconds": int,
      "result_time": "HH:MM:SS",
      "days_offset": int
    }

    Examples
    --------
    time_math("add", time="09:20", delta="1:45") -> 11:05:00
    time_math("sub", time="00:10", delta="15m")  -> 23:55:00, days_offset=-1
    time_math("add", time="23:00", hours=3)      -> 02:00:00, days_offset=+1
    """
    base_sec = _parse_hms_time(time)
    dur_sec = _parse_duration(delta, hours, minutes, seconds)
    if op == "sub":
        dur_sec = -dur_sec
    total = base_sec + dur_sec

    input_time_norm, _ = _format_hms(base_sec, wrap_24h=True)  # normalize display
    result_str, days = _format_hms(total, wrap_24h=wrap_24h)

    return {
        "input_time": input_time_norm,
        "delta_seconds": dur_sec,
        "result_time": result_str,
        "days_offset": days
    }

if __name__ == "__main__":
    print("MCP server starting... waiting for Inspector")
    mcp.run()
