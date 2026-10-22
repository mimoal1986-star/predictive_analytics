import copy
import gc
import io
import math
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

# ===============================================================
# СХЕМА ДАННЫХ
# ===============================================================
REQUIRED_COLUMNS = [
    "Тип файла",
    "Год",
    "Месяц",
    "Клиент",
    "Код проекта",
    "Волна",
    "ASM",
    "Регион short",
    "RS",
    "RS ID",
    "Дата визита",
]

FINAL_COLUMNS = REQUIRED_COLUMNS

CATEGORICAL_COLUMNS = [
    "Тип файла", "Клиент", "Код проекта", "Волна", "ASM", "Регион short",
]

PARQUET_PATH = Path("data/report_data.parquet")
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)


def validate_parquet(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)


def compute_metrics(df: pd.DataFrame, **params) -> dict:
    return {}


# ===============================================================
# АЛИАСЫ КОЛОНОК
# ===============================================================
DEFAULT_ZP_ALIASES = {
    "ASM": ["ACC", "АСС", "ASM", "АСМ"],
    "Дата визита": ["FinishTime", "Дата визита"],
    "Волна": ["SetName", "Проекты", "Волна"],
    "Код проекта": ["SetCode"],
    "Клиент": ["ClientName"],
    "Регион short": [
        "RegionName", "Re", "Регион", "RegionName Чек",
        "Re Чеккер", "РЕГИОН", "регион",
    ],
    "RS": ["Тайный покупатель Fullname", "RS"],
    "RS ID": ["Тайный покупатель ID"],
}

DEFAULT_HVOSTY_ALIASES = {
    "ASM": DEFAULT_ZP_ALIASES["ASM"],
    "Дата визита": [],
    "Волна": ["Название волны", "SetName"],
    "Код проекта": ["Код проекта"] + DEFAULT_ZP_ALIASES["Код проекта"],
    "Клиент": ["Проект", "Проекты"] + DEFAULT_ZP_ALIASES["Клиент"],
    "Регион short": ["Регион"] + DEFAULT_ZP_ALIASES["Регион short"],
    "RS": [],
    "RS ID": [],
}

TARGET_LABELS = {
    "ASM": "АСМ",
    "Дата визита": "Дата визита",
    "Волна": "Волна",
    "Код проекта": "Код проекта",
    "Клиент": "Клиент",
    "Регион short": "Регион short",
    "RS": "РС",
    "RS ID": "RS ID",
}


# ===============================================================
# ПАРСЕР ДАТ
# ===============================================================
EXCEL_EPOCH = datetime(1899, 12, 30)
SERIAL_MIN = 1
SERIAL_MAX = 100_000

RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

EN_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

_RE_ISO = re.compile(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})(?:[ T].*)?$")
_RE_DMY = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})(?:[ T].*)?$")
_RE_DM = re.compile(r"(\d{1,2})[./-](\d{1,2})")
_RE_RU = re.compile(r"(\d{1,2})\s+([а-яё]+)(?:\s+(\d{2,4}))?")
_RE_EN_DMY = re.compile(r"(\d{1,2})[\s-]+([a-z]+)\.?[\s-]*(\d{2,4})?")
_RE_EN_MDY = re.compile(r"([a-z]+)\.?[\s-]+(\d{1,2})(?:,?[\s-]+(\d{2,4}))?")


def _safe_ts(year: int, month: int, day: int) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except (ValueError, OverflowError):
        return None


def _resolve_year(yy: int) -> int:
    return 2000 + yy if yy < 70 else 1900 + yy


def _from_serial(num: float) -> pd.Timestamp | None:
    if not (SERIAL_MIN <= num <= SERIAL_MAX):
        return None
    try:
        return pd.Timestamp(EXCEL_EPOCH + timedelta(days=float(num)))
    except (OverflowError, ValueError):
        return None


def parse_visit_date(raw_value, sheet_year: int | None = None) -> pd.Timestamp | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, float):
        if math.isnan(raw_value):
            return None
        return _from_serial(raw_value)
    if isinstance(raw_value, (int, bool)):
        return _from_serial(float(raw_value))
    if isinstance(raw_value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(raw_value)

    text = str(raw_value).strip()
    if not text:
        return None

    if text.isdigit():
        if len(text) == 8:
            dd, mm, yyyy = int(text[:2]), int(text[2:4]), int(text[4:])
            ts = _safe_ts(yyyy, mm, dd)
            if ts is not None:
                return ts
            yyyy, mm, dd = int(text[:4]), int(text[4:6]), int(text[6:])
            return _safe_ts(yyyy, mm, dd)
        if len(text) == 6:
            dd, mm, yy = int(text[:2]), int(text[2:4]), int(text[4:])
            return _safe_ts(_resolve_year(yy), mm, dd)
        return _from_serial(float(text))

    m = _RE_ISO.fullmatch(text)
    if m:
        return _safe_ts(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = _RE_DMY.fullmatch(text)
    if m:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = yy if yy >= 100 else _resolve_year(yy)
        return _safe_ts(year, mm, dd)

    m = _RE_DM.fullmatch(text)
    if m:
        if sheet_year is None:
            return None
        return _safe_ts(sheet_year, int(m.group(2)), int(m.group(1)))

    m = _RE_RU.fullmatch(text.lower())
    if m:
        dd = int(m.group(1))
        mm = RU_MONTHS.get(m.group(2))
        if mm is None:
            return None
        if m.group(3):
            yy = int(m.group(3))
            year = yy if yy >= 100 else _resolve_year(yy)
        elif sheet_year is not None:
            year = sheet_year
        else:
            return None
        return _safe_ts(year, mm, dd)

    m = _RE_EN_DMY.fullmatch(text.lower())
    if m:
        dd = int(m.group(1))
        mm = EN_MONTHS.get(m.group(2).rstrip("."))
        if mm is None:
            return None
        if m.group(3):
            yy = int(m.group(3))
            year = yy if yy >= 100 else _resolve_year(yy)
        elif sheet_year is not None:
            year = sheet_year
        else:
            return None
        return _safe_ts(year, mm, dd)

    m = _RE_EN_MDY.fullmatch(text.lower())
    if m:
        mm = EN_MONTHS.get(m.group(1).rstrip("."))
        if mm is None:
            return None
        dd = int(m.group(2))
        if m.group(3):
            yy = int(m.group(3))
            year = yy if yy >= 100 else _resolve_year(yy)
        elif sheet_year is not None:
            year = sheet_year
        else:
            return None
        return _safe_ts(year, mm, dd)

    return None


def parse_dates_vectorized(series: pd.Series, sheet_year: int) -> pd.Series:
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    normalized = series.copy()
    is_str_mask = normalized.apply(lambda v: isinstance(v, str))
    if is_str_mask.any():
        s = normalized.loc[is_str_mask].astype(str).str.strip()
        s = s.str.replace(r"^(\d+),(\d+)$", r"\1.\2", regex=True)
        normalized.loc[is_str_mask] = s

    numeric = pd.to_numeric(normalized, errors="coerce")
    is_num = numeric.notna() & (numeric >= SERIAL_MIN) & (numeric <= SERIAL_MAX)
    if is_num.any():
        result.loc[is_num] = pd.Timestamp(EXCEL_EPOCH) + pd.to_timedelta(
            numeric[is_num].astype("float64").astype("int64"), unit="D"
        )

    is_str = series.notna() & ~is_num
    if is_str.any():
        str_vals = series[is_str].astype(str).str.strip()

        iso_mask = str_vals.str.match(
            r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$", na=False
        )
        if iso_mask.any():
            iso_dates = pd.to_datetime(
                str_vals[iso_mask].str.extract(
                    r"^(\d{4}-\d{1,2}-\d{1,2})", expand=False
                ),
                format="%Y-%m-%d",
                errors="coerce",
            )
            result.loc[str_vals.index[iso_mask]] = iso_dates.values

        rest_mask = ~iso_mask
        if rest_mask.any():
            rest_vals = str_vals[rest_mask]
            rest_vals = rest_vals.str.replace(
                r"[ T]\d{1,2}:\d{2}(?::\d{2})?.*$", "", regex=True
            )
            parsed = pd.to_datetime(rest_vals, dayfirst=True, errors="coerce")
            result.loc[rest_vals.index] = parsed.values

    unresolved = result.isna() & series.notna()
    if unresolved.any():
        extra = series[unresolved].apply(
            lambda v: parse_visit_date(v, sheet_year=sheet_year)
        )
        result.loc[unresolved] = pd.to_datetime(extra, errors="coerce")

    return pd.to_datetime(result, errors="coerce")


# ===============================================================
# ЧТЕНИЕ EXCEL
# ===============================================================
def _open_excel(path: str) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(path, engine="calamine")
    except Exception:
        return pd.ExcelFile(path, engine="openpyxl")


def _resolve_columns(
    df: pd.DataFrame, aliases: dict[str, list[str]]
) -> dict[str, str]:
    mapping = {}
    for target, names in aliases.items():
        for name in names:
            if name in df.columns:
                mapping[target] = name
                break
    return mapping


def _parse_sheet(
    xls: pd.ExcelFile,
    sheet_name: str,
    sheet_year: int,
    sheet_month: int,
    aliases: dict[str, list[str]],
    file_type: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        df = xls.parse(sheet_name)
    except Exception as e:
        st.warning(f"[{file_type}] Лист «{sheet_name}» не прочитан: {e}")
        return None, None

    df.columns = [str(c).strip() for c in df.columns]

    mapping = _resolve_columns(df, aliases)
    missing = [
        t for t in aliases
        if t not in mapping and aliases[t]
    ]
    if missing:
        lines = [
            f"[{file_type}] Лист «{sheet_name}» "
            f"{TARGET_LABELS[t]} = пусто: нет колонок ['{t}']"
            for t in missing
        ]
        st.warning("\n".join(lines))

    out = pd.DataFrame(index=df.index)
    for target in aliases:
        if target in mapping:
            out[target] = df[mapping[target]]
        else:
            out[target] = pd.NA

    raw_finish = (
        df[mapping["Дата визита"]]
        if "Дата визита" in mapping
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    crit_id = df["CritID"] if "CritID" in df.columns else None

    is_zp = file_type == "ЗП"

    if not is_zp:
        out["Дата визита"] = pd.NaT
        bad_mask = pd.Series(False, index=df.index)
    elif "Дата визита" not in mapping:
        out["Дата визита"] = pd.Timestamp(
            year=sheet_year, month=sheet_month, day=1
        )
        bad_mask = pd.Series(False, index=df.index)
    else:
        out["Дата визита"] = pd.to_datetime(
            parse_dates_vectorized(out["Дата визита"], sheet_year),
            errors="coerce",
        )
        out["Дата визита"] = out["Дата визита"].dt.normalize()
        bad_mask = out["Дата визита"].isna()
        fallback_date = pd.Timestamp(
            year=sheet_year, month=sheet_month, day=1
        )
        out.loc[bad_mask, "Дата визита"] = fallback_date

    good_rows = out[~bad_mask]
    bad_rows = out[bad_mask]

    bad_report = None
    if "Дата визита" in mapping and not bad_rows.empty:
        bad_report = pd.DataFrame({
            "Тип файла": file_type,
            "Вкладка": sheet_name,
            "FinishTime (оригинал)": raw_finish.loc[bad_rows.index].values,
            "CritID": (
                crit_id.loc[bad_rows.index].values
                if crit_id is not None
                else None
            ),
        })

    return good_rows, bad_report


def _parse_sheet_name_zp(sheet_name: str) -> tuple[int, int] | None:
    name = sheet_name.strip()
    if len(name) != 4 or not name.isdigit():
        return None
    month = int(name[:2])
    year = 2000 + int(name[2:])
    if not (1 <= month <= 12):
        return None
    return year, month


def _parse_sheet_name_hvosty(sheet_name: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"Хвосты[\s_]+(\d{2})\.(\d{4})", sheet_name.strip())
    if not m:
        return None
    month = int(m.group(1))
    year = int(m.group(2))
    if not (1 <= month <= 12):
        return None
    if (year, month) < (2024, 5):
        return None
    return year, month


# ===============================================================
# ОБРАБОТКА ФАЙЛА
# ===============================================================
def process_file(
    excel_path: str,
    aliases: dict[str, list[str]],
    file_type: str,
    parse_name,
    progress_callback=None,
) -> tuple[
    list[pd.DataFrame], list[pd.DataFrame], list[str], int, int,
    list[str], list[str],
]:
    xls = _open_excel(excel_path)
    sheet_names = xls.sheet_names
    if not sheet_names:
        raise ValueError(f"[{file_type}] В файле нет вкладок")

    frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    used_sheet_names: list[str] = []
    total = len(sheet_names)
    used = 0

    for i, sheet_name in enumerate(sheet_names, start=1):
        if progress_callback:
            progress_callback(i, total, sheet_name, file_type)

        ym = parse_name(sheet_name)
        if ym is None:
            skipped.append(f"[{file_type}] {sheet_name}")
            continue

        year, month = ym
        good, bad = _parse_sheet(
            xls, sheet_name, year, month, aliases, file_type
        )
        if good is None:
            skipped.append(f"[{file_type}] {sheet_name}")
            continue

        used += 1
        used_sheet_names.append(sheet_name)

        if not good.empty:
            good = good.copy()
            good.insert(0, "Тип файла", file_type)
            good.insert(1, "Год", year)
            good.insert(2, "Месяц", month)
            for c in FINAL_COLUMNS:
                if c not in good.columns:
                    good[c] = pd.NA
            good = good[FINAL_COLUMNS]
            frames.append(good)

        if bad is not None and not bad.empty:
            error_frames.append(bad)

        del good, bad
        gc.collect()

    return (
        frames, error_frames, skipped, total, used,
        used_sheet_names, sheet_names,
    )


# ===============================================================
# КОНВЕРТАЦИЯ
# ===============================================================
def convert_to_parquet(
    zp_path: str | None,
    hvosty_path: str | None,
    parquet_path: Path,
    zp_aliases: dict[str, list[str]],
    hvosty_aliases: dict[str, list[str]],
    progress_callback=None,
) -> dict:
    all_frames: list[pd.DataFrame] = []
    all_errors: list[pd.DataFrame] = []
    all_skipped: list[str] = []
    total_sheets = 0
    used_sheets = 0
    hvosty_excluded_names: list[str] = []
    hvosty_included_names: list[str] = []

    if zp_path:
        frames, errors, skipped, total, used, _, _ = process_file(
            zp_path, zp_aliases, "ЗП",
            _parse_sheet_name_zp, progress_callback,
        )
        all_frames.extend(frames)
        all_errors.extend(errors)
        all_skipped.extend(skipped)
        total_sheets += total
        used_sheets += used

    if hvosty_path:
        (
            frames, errors, skipped, total, used,
            used_names, hvosty_sheet_names,
        ) = process_file(
            hvosty_path, hvosty_aliases, "Хвосты",
            _parse_sheet_name_hvosty, progress_callback,
        )
        all_frames.extend(frames)
        all_errors.extend(errors)
        all_skipped.extend(skipped)
        total_sheets += total
        used_sheets += used

        hvosty_included_names = list(used_names)
        used_set = set(used_names)
        hvosty_excluded_names = [
            n for n in hvosty_sheet_names if n not in used_set
        ]

    if not all_frames:
        raise ValueError(
            "Не удалось прочитать ни одной вкладки. "
            f"Пропущены: {all_skipped}"
        )

    combined = pd.concat(all_frames, ignore_index=True)
    del all_frames
    gc.collect()

    combined = combined[FINAL_COLUMNS]

    text_cols = [
        c for c in FINAL_COLUMNS
        if c not in ("Год", "Месяц", "Дата визита", "RS ID")
    ]
    for col in text_cols:
        combined[col] = (
            combined[col]
            .astype("string")
            .str.replace("\u00a0", " ", regex=False)
            .str.strip()
        )

    combined["Год"] = combined["Год"].astype("int16")
    combined["Месяц"] = combined["Месяц"].astype("int8")

    for col in CATEGORICAL_COLUMNS:
        combined[col] = combined[col].astype("category")

    rs_id = pd.to_numeric(combined["RS ID"], errors="coerce")
    bad_rs = rs_id.notna() & (rs_id % 1 != 0)
    if bad_rs.any():
        st.warning(f"RS ID: {bad_rs.sum()} дробных значений — округлены")
    combined["RS ID"] = rs_id.round().astype("Int64")

    combined.to_parquet(parquet_path, index=False, compression="zstd")

    error_report = pd.concat(all_errors, ignore_index=True) if all_errors else None

    stats = {
        "rows": len(combined),
        "columns": len(combined.columns),
        "sheets_used": used_sheets,
        "sheets_total": total_sheets,
        "skipped": all_skipped,
        "parquet_size_mb": parquet_path.stat().st_size / (1024 * 1024),
        "bad_dates": len(error_report) if error_report is not None else 0,
        "error_report": error_report,
        "hvosty_excluded_names": hvosty_excluded_names,
        "hvosty_included_names": hvosty_included_names,
    }
    return stats


# ===============================================================
# КЭШИРУЕМОЕ ЧТЕНИЕ PARQUET
# ===============================================================
@st.cache_data(show_spinner=False)
def _read_parquet_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(file_bytes))


# ===============================================================
# ИНИЦИАЛИЗАЦИЯ SESSION_STATE
# ===============================================================
if "convert" not in st.session_state:
    st.session_state.convert = {
        "zp_bytes": None,
        "hvosty_bytes": None,
        "stats": None,
        "parquet_bytes": None,
        "bad_dates_bytes": None,
        "sheets_bytes": None,
    }

if "dataset" not in st.session_state:
    st.session_state.dataset = {
        "df": None,
        "loaded_at": None,
        "metrics": None,
        "file_hash": None,
    }

if "report" not in st.session_state:
    st.session_state.report = None

if "aliases" not in st.session_state:
    st.session_state.aliases = {
        "zp": copy.deepcopy(DEFAULT_ZP_ALIASES),
        "hvosty": copy.deepcopy(DEFAULT_HVOSTY_ALIASES),
    }


# ===============================================================
# UI
# ===============================================================
st.set_page_config(page_title="Аналитика визитов", layout="wide")
st.title("Аналитика визитов")

tab_upload, tab_reports, tab_settings = st.tabs(
    ["Загрузка данных", "Отчеты", "Настройки проекта"]
)


# ---------------------------------------------------------------
# ВКЛАДКА «ЗАГРУЗКА ДАННЫХ»
# ---------------------------------------------------------------
with tab_upload:
    st.header("Загрузка данных")
    st.caption(
        "Загрузите Parquet, сформированный на вкладке "
        "«Настройки проекта». Проверяются обязательные колонки."
    )

    parquet_file = st.file_uploader(
        "Parquet-файл (report_data.parquet)",
        type=["parquet"],
        key="upload_parquet",
    )

    ok = False
    missing: list[str] = []
    df_loaded = None

    if parquet_file is not None:
        file_bytes = parquet_file.getvalue()
        try:
            df_loaded = _read_parquet_cached(file_bytes)
        except Exception as e:
            st.error(f"Не удалось прочитать Parquet: {e}")
            df_loaded = None

        if df_loaded is not None:
            ok, missing = validate_parquet(df_loaded)
            if not ok:
                st.error("Колонки отсутствуют: " + ", ".join(missing))
            else:
                st.success(
                    f"Parquet прочитан: {len(df_loaded):,} строк, "
                    f"{len(df_loaded.columns)} колонок"
                )

    if st.button(
        "Загрузить",
        type="primary",
        disabled=not ok,
    ):
        st.session_state.dataset["df"] = df_loaded
        st.session_state.dataset["loaded_at"] = datetime.now()
        st.session_state.dataset["file_hash"] = hash(
            parquet_file.getvalue()
        )
        st.session_state.dataset["metrics"] = None
        st.session_state.report = None
        st.success("Датасет загружен. Перейдите на вкладку «Отчеты».")


# ---------------------------------------------------------------
# ВКЛАДКА «ОТЧЕТЫ»
# ---------------------------------------------------------------
with tab_reports:
    st.header("Отчеты")

    rep_clients, rep_regions, rep_dsm, rep_summary = st.tabs(
        ["Клиенты", "Регионы", "DSM", "Сводный"]
    )

    # -----------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ОТЧЁТОВ
    # -----------------------------------------------------------
    def _month_label(year: int, month: int) -> str:
        return f"{month}_{year % 100:02d}"

    def compute_kpi(
        df_client_all: pd.DataFrame,
        months: list[tuple[int, int]],
    ) -> dict:
        plans = []
        facts = []
        rs_counts = []

        for y, m in months:
            sub = df_client_all[
                (df_client_all["Год"] == y)
                & (df_client_all["Месяц"] == m)
            ]
            if sub.empty:
                continue
            plans.append(len(sub))
            facts.append(int((sub["Тип файла"] == "ЗП").sum()))
            rs_counts.append(sub["RS"].dropna().nunique())

        n = len(plans)
        if n == 0:
            return {
                "plan_avg": 0,
                "fact_avg": 0,
                "pf_percent": 0,
                "rs_count_avg": 0,
                "ankets_per_rs": 0,
            }

        plan_avg = round(sum(plans) / n)
        fact_avg = round(sum(facts) / n)
        rs_avg = round(sum(rs_counts) / n)
        pf_percent = (
            round(fact_avg / plan_avg * 100) if plan_avg > 0 else 0
        )
        ankets = (
            round(fact_avg / rs_avg) if rs_avg > 0 else 0
        )

        return {
            "plan_avg": plan_avg,
            "fact_avg": fact_avg,
            "pf_percent": pf_percent,
            "rs_count_avg": rs_avg,
            "ankets_per_rs": ankets,
        }

    def compute_monthly_table(
        df_client_all: pd.DataFrame,
        months: list[tuple[int, int]],
    ) -> pd.DataFrame:
        rows = {
            "План проекта, шт": [],
            "Факт проекта, шт": [],
            "ПФ клиент, %": [],
            "Кол-во RS, чел.": [],
            "Кол-во анкет на 1 RS, шт.": [],
        }
        for y, m in reversed(months):
            sub = df_client_all[
                (df_client_all["Год"] == y)
                & (df_client_all["Месяц"] == m)
            ]
            if sub.empty:
                rows["План проекта, шт"].append(None)
                rows["Факт проекта, шт"].append(None)
                rows["ПФ клиент, %"].append(None)
                rows["Кол-во RS, чел."].append(None)
                rows["Кол-во анкет на 1 RS, шт."].append(None)
            else:
                plan = len(sub)
                fact = int((sub["Тип файла"] == "ЗП").sum())
                pf = (fact / plan * 100) if plan > 0 else 0.0
                rs_unique = sub["RS"].dropna().nunique()
                ankets = (fact / rs_unique) if rs_unique > 0 else 0.0
                rows["План проекта, шт"].append(plan)
                rows["Факт проекта, шт"].append(fact)
                rows["ПФ клиент, %"].append(round(pf))
                rows["Кол-во RS, чел."].append(rs_unique)
                rows["Кол-во анкет на 1 RS, шт."].append(round(ankets))

        columns = [
            _month_label(y, m) for y, m in reversed(months)
        ]
        table = pd.DataFrame(rows, index=columns).T
        table.index.name = "Метрика"
        return table

    # -----------------------------------------------------------
    # ПОДВКЛАДКА «КЛИЕНТЫ»
    # -----------------------------------------------------------
    with rep_clients:
        df_dataset = st.session_state.dataset.get("df")

        if df_dataset is None:
            st.info(
                "Сначала загрузите Parquet на вкладке "
                "«Загрузка данных»."
            )
        else:
            clients = sorted(
                df_dataset["Клиент"]
                .dropna()
                .unique()
                .astype(str)
                .tolist()
            )

            if not clients:
                st.warning("В датасете нет клиентов.")
            else:
                client = st.selectbox(
                    "Клиент",
                    options=clients,
                    key="rep_clients_client",
                )

                df_client_all = df_dataset[
                    df_dataset["Клиент"] == client
                ]

                st.markdown("### 📊 Ключевые показатели")
                report_key = (client,)

                if st.button("Рассчитать", type="primary"):
                    if df_client_all.empty:
                        st.session_state.report = {
                            "key": report_key,
                            "empty": True,
                            "kpi": None,
                            "table": None,
                        }
                    else:
                        pairs = sorted(
                            set(
                                (int(y), int(m))
                                for y, m in df_client_all[
                                    ["Год", "Месяц"]
                                ]
                                .dropna()
                                .itertuples(index=False, name=None)
                            )
                        )
                        if pairs:
                            y_min, m_min = pairs[0]
                            y_max, m_max = pairs[-1]
                            months = []
                            y, m = y_min, m_min
                            while (y, m) <= (y_max, m_max):
                                months.append((y, m))
                                if m == 12:
                                    y, m = y + 1, 1
                                else:
                                    m += 1
                        else:
                            months = []

                        kpi = compute_kpi(df_client_all, months)

                        table = compute_monthly_table(
                            df_client_all, months
                        )
                        st.session_state.report = {
                            "key": report_key,
                            "empty": False,
                            "kpi": kpi,
                            "table": table,
                        }

                report = st.session_state.report
                if report is None or report["key"] != report_key:
                    st.info(
                        "Нажмите «Рассчитать», чтобы построить отчёт."
                    )
                elif report["empty"]:
                    st.info("Нет данных по выбранному клиенту.")
                else:
                    kpi = report["kpi"]
                    table = report["table"]

                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric(
                        "📊 План ср., шт",
                        f"{kpi['plan_avg']:,}",
                    )
                    k2.metric(
                        "✅ Факт ср., шт",
                        f"{kpi['fact_avg']:,}",
                    )
                    k3.metric(
                        "🎯 ПФ ср., %",
                        f"{kpi['pf_percent']}%",
                    )
                    k4.metric(
                        "👥 Кол-во RS ср., чел.",
                        f"{kpi['rs_count_avg']:,}",
                    )
                    k5.metric(
                        "📋 Кол-во анкет на 1 RS ср., шт.",
                        f"{kpi['ankets_per_rs']}",
                    )

                    st.markdown(
                        "### 📈 Ключевые показатели по месяцам"
                    )
                    st.dataframe(
                        table,
                        use_container_width=True,
                    )

                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(
                        excel_buffer, engine="openpyxl"
                    ) as writer:
                        table.to_excel(
                            writer,
                            sheet_name="Ключевые показатели",
                        )
                    st.download_button(
                        "📥 Выгрузить в Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"kpi_{client}.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                    )

                    dataset_buffer = io.BytesIO()
                    with pd.ExcelWriter(
                        dataset_buffer, engine="openpyxl"
                    ) as writer:
                        df_client_all.to_excel(
                            writer,
                            index=False,
                            sheet_name="Датасет",
                        )
                    st.download_button(
                        "📥 Выгрузить датасет",
                        data=dataset_buffer.getvalue(),
                        file_name=f"dataset_{client}.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                    )

                    st.markdown("### 📊 Дашборд")
                    st.markdown("<br>", unsafe_allow_html=True)

                    font = (
                        "Inter, -apple-system, BlinkMacSystemFont, "
                        "'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
                    )

                    def render_bar_chart(
                        df: pd.DataFrame, 
                        metric_name: str, 
                        title: str, 
                        y_title: str, 
                        color_rule
                    ):
                        """Универсальная функция для отрисовки столбиковых графиков"""
                        series = df.loc[metric_name]
                        chart_data = pd.DataFrame({
                            "Месяц": series.index,
                            "Value": series.values
                        }).dropna()

                        if not chart_data.empty:
                            chart_data["Value"] = chart_data["Value"].astype(int)
                            chart_data = chart_data.iloc[::-1].reset_index(drop=True)

                            # Вызов функции цвета
                            chart_data["Цвет"] = chart_data["Value"].apply(color_rule)

                            # Высота оси Y = макс значение + 20%, но не меньше 1
                            max_val = chart_data["Value"].max()
                            y_max = max(max_val * 1.2, 1)

                            bars = alt.Chart(chart_data).mark_bar(
                                cornerRadiusTopLeft=2,
                                cornerRadiusTopRight=2,
                                size=40 # Фиксированная ширина столбца
                            ).encode(
                                x=alt.X(
                                    "Месяц:N",
                                    sort=None,
                                    scale=alt.Scale(paddingInner=0.2, paddingOuter=0.5),
                                    axis=alt.Axis(
                                        labelAngle=0,
                                        labelFont=font,
                                        title=None,
                                    ),
                                ),
                                y=alt.Y(
                                    "Value:Q",
                                    scale=alt.Scale(domain=[0, y_max]),
                                    axis=alt.Axis(
                                        format="d",
                                        title=y_title,
                                        labelFont=font,
                                        titleFont=font,
                                    ),
                                ),
                                color=alt.Color(
                                    "Цвет:N", scale=None, legend=None
                                ),
                            )

                            text = bars.mark_text(
                                align="center",
                                baseline="bottom",
                                dy=-5,
                                font=font,
                                fontSize=12,
                                color="#333333",
                            ).encode(
                                text=alt.Text("Value:Q", format="d")
                            )

                            # Считаем ширину: 40px на столбец + 40px на отступы + 60px на ось Y
                            calc_width = len(chart_data) * 80 + 60

                            chart = (bars + text).properties(
                                title=alt.TitleParams(
                                    text=title,
                                    font=font,
                                    fontSize=16,
                                    anchor="start",
                                ),
                                height=400,
                                width=calc_width # Применяем ширину
                            ).configure_view(
                                strokeWidth=0
                            ).configure_axis(
                                grid=False
                            )

                            # Отключаем растягивание на всю ширину
                            st.altair_chart(chart, use_container_width=False)

                    def get_quantile_colors(series, top_is_violet=True):
                        """
                        Возвращает функцию для apply, которая красит
                        топ 10% в фиолетовый (или фуксию), антитоп 10% в фуксию (или фиолетовый).
                        Если все значения равны, возвращает серый.
                        """
                        clean_series = series.dropna()
                        if clean_series.empty:
                            return lambda x: "#757575"
                        
                        # Если все значения одинаковые, красим всё в серый
                        if clean_series.min() == clean_series.max():
                            return lambda x: "#757575"

                        # Считаем квантили
                        q10 = clean_series.quantile(0.1)
                        q90 = clean_series.quantile(0.9)

                        def color_func(val):
                            if val >= q90:
                                return "#6A1B9A" if top_is_violet else "#F50057"
                            elif val <= q10:
                                return "#F50057" if top_is_violet else "#6A1B9A"
                            else:
                                return "#757575"
                        return color_func

                    # 1. ПФ клиент, %
                    def pf_color(val):
                        if val == 100:
                            return "#6A1B9A"
                        elif val < 80:
                            return "#F50057"
                        else:
                            return "#757575"

                    render_bar_chart(
                        table, 
                        "ПФ клиент, %", 
                        "ПФ клиент, %", 
                        "ПФ клиент, %", 
                        pf_color
                    )

                    # 2. Кол-во RS, чел.
                    # Топ 10% - фиолетовый, антитоп 10% - фуксия
                    rs_series = table.loc["Кол-во RS, чел."]
                    rs_color = get_quantile_colors(rs_series, top_is_violet=True)

                    render_bar_chart(
                        table, 
                        "Кол-во RS, чел.", 
                        "Кол-во RS, чел.", 
                        "Кол-во RS, чел.", 
                        rs_color
                    )

                    # 3. Кол-во анкет на 1 RS, шт.
                    # Антитоп 10% (самые высокие) - фиолетовый, топ 10% - фуксия
                    ankets_series = table.loc["Кол-во анкет на 1 RS, шт."]
                    ankets_color = get_quantile_colors(ankets_series, top_is_violet=False)

                    render_bar_chart(
                        table, 
                        "Кол-во анкет на 1 RS, шт.", 
                        "Кол-во анкет на 1 RS, шт.", 
                        "Кол-во анкет на 1 RS, шт.", 
                        ankets_color
                    )

    # -----------------------------------------------------------
    # ПОДВКЛАДКА «РЕГИОНЫ» (заглушка)
    # -----------------------------------------------------------
    with rep_regions:
        st.info("В разработке.")

    # -----------------------------------------------------------
    # ПОДВКЛАДКА «DSM» (заглушка)
    # -----------------------------------------------------------
    with rep_dsm:
        st.info("В разработке.")

    # -----------------------------------------------------------
    # ПОДВКЛАДКА «СВОДНЫЙ» (заглушка)
    # -----------------------------------------------------------
    with rep_summary:
        st.info("В разработке.")


# ---------------------------------------------------------------
# ВКЛАДКА «НАСТРОЙКИ ПРОЕКТА»
# ---------------------------------------------------------------
with tab_settings:
    st.header("Настройки проекта")
    st.caption(
        "Конвертация Excel (ЗП + Хвосты) → Parquet. "
        "Выполняется раз в месяц, результат сохраняется локально."
    )

    zp_file = st.file_uploader(
        "Файл ЗП (вкладки в формате ММГГ, например 0125)",
        type=["xlsx", "xlsm"],
        key="settings_zp",
    )
    hvosty_file = st.file_uploader(
        "Файл Хвосты (вкладки в формате 'Хвосты 09.2026')",
        type=["xlsx", "xlsm"],
        key="settings_hvosty",
    )

    if zp_file is not None:
        st.session_state.convert["zp_bytes"] = zp_file.getvalue()
    if hvosty_file is not None:
        st.session_state.convert["hvosty_bytes"] = hvosty_file.getvalue()

    if st.button("Конвертировать в Parquet", type="primary"):
        conv = st.session_state.convert
        if conv["zp_bytes"] is None and conv["hvosty_bytes"] is None:
            st.error("Загрузите хотя бы один файл: ЗП или Хвосты.")
        else:
            zp_path = None
            hvosty_path = None
            tmp_paths: list[str] = []

            try:
                if conv["zp_bytes"] is not None:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".xlsx"
                    ) as tmp:
                        tmp.write(conv["zp_bytes"])
                        zp_path = tmp.name
                        tmp_paths.append(zp_path)

                if conv["hvosty_bytes"] is not None:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".xlsx"
                    ) as tmp:
                        tmp.write(conv["hvosty_bytes"])
                        hvosty_path = tmp.name
                        tmp_paths.append(hvosty_path)

                progress = st.progress(0.0, text="Читаю вкладки…")

                def _cb(i, total, sheet_name, file_type):
                    progress.progress(
                        min(i / total, 1.0),
                        text=f"[{file_type}] {i}/{total}: {sheet_name}",
                    )

                try:
                    with st.spinner("Конвертирую…"):
                        stats = convert_to_parquet(
                            zp_path, hvosty_path, PARQUET_PATH,
                            st.session_state.aliases["zp"],
                            st.session_state.aliases["hvosty"],
                            progress_callback=_cb,
                        )
                except ValueError as e:
                    progress.empty()
                    st.error(f"Конвертация не удалась: {e}")
                    st.stop()
                progress.empty()

                conv["stats"] = stats
                conv["parquet_bytes"] = PARQUET_PATH.read_bytes()

                if stats["bad_dates"] > 0:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(
                        buffer, engine="openpyxl"
                    ) as writer:
                        stats["error_report"].to_excel(
                            writer, index=False, sheet_name="BadDates"
                        )
                    conv["bad_dates_bytes"] = buffer.getvalue()
                else:
                    conv["bad_dates_bytes"] = None

                if (
                    stats["hvosty_included_names"]
                    or stats["hvosty_excluded_names"]
                ):
                    sheets_buffer = io.BytesIO()
                    with pd.ExcelWriter(
                        sheets_buffer, engine="openpyxl"
                    ) as writer:
                        if stats["hvosty_included_names"]:
                            pd.DataFrame({
                                "Хвосты добавлены":
                                    stats["hvosty_included_names"],
                            }).to_excel(
                                writer, index=False, sheet_name="Included"
                            )
                        if stats["hvosty_excluded_names"]:
                            pd.DataFrame({
                                "Вкладки Хвост исключены":
                                    stats["hvosty_excluded_names"],
                            }).to_excel(
                                writer, index=False, sheet_name="Excluded"
                            )
                    conv["sheets_bytes"] = sheets_buffer.getvalue()
                else:
                    conv["sheets_bytes"] = None

            finally:
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    conv = st.session_state.convert
    if conv["stats"] is not None:
        stats = conv["stats"]

        st.success("Готово!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Строк", f"{stats['rows']:,}")
        col2.metric(
            "Вкладок обработано",
            f"{stats['sheets_used']} / {stats['sheets_total']}",
        )
        col3.metric(
            "Размер Parquet, МБ",
            f"{stats['parquet_size_mb']:.2f}",
        )

        if conv["parquet_bytes"]:
            st.download_button(
                "Скачать Parquet",
                data=conv["parquet_bytes"],
                file_name="report_data.parquet",
                mime="application/octet-stream",
            )

        if conv["bad_dates_bytes"]:
            st.warning(
                f"Не распознано дат визита: {stats['bad_dates']}. "
                "Скачайте отчёт ниже."
            )
            st.download_button(
                "Скачать Excel с ошибками дат",
                data=conv["bad_dates_bytes"],
                file_name="bad_dates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if conv["sheets_bytes"]:
            st.download_button(
                "Скачать Excel со списком вкладок Хвостов (Included/Excluded)",
                data=conv["sheets_bytes"],
                file_name="hvosty_sheets.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
