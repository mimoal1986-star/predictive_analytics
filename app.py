import gc
import io
import math
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Алиасы колонок: для каждой целевой — список возможных исходных
# ---------------------------------------------------------------
COLUMN_ALIASES = {
    # ASM: латиница ACC/ASM, кириллица АСС/АСМ
    "ASM": ["ACC", "АСС", "ASM", "АСМ"],
    # Дата визита: англ. FinishTime, рус. «Дата визита»
    "Дата визита": ["FinishTime", "Дата визита"],
    # Волна: SetName, а также рус. «Проекты», «Волна»
    "Волна": ["SetName", "Проекты", "Волна"],
    "Код проекта": ["SetCode"],
    "Клиент": ["ClientName"],
    # Регион: RegionName, Re, Регион, RegionName Чек, Re Чеккер, РЕГИОН
    "Регион short": [
        "RegionName", "Re", "Регион", "RegionName Чек",
        "Re Чеккер", "РЕГИОН",
    ],
    # RS: Тайный покупатель Fullname, RS
    "RS": ["Тайный покупатель Fullname", "RS"],
    "RS ID": ["Тайный покупатель ID"],
}

# Человекочитаемые названия для сообщений
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

FINAL_COLUMNS = [
    "Год",
    "Месяц",
    "Дата визита",
    "ASM",
    "Волна",
    "Код проекта",
    "Клиент",
    "Регион short",
    "RS",
    "RS ID",
]

CATEGORICAL_COLUMNS = ["ASM", "Волна", "Код проекта", "Клиент", "Регион short"]

PARQUET_PATH = Path("data/report_data.parquet")
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Парсер дат: прекомпилированные регулярки + векторизация
# ---------------------------------------------------------------
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

    # 1. Числа → Excel serial (дробную часть отбрасываем — это время)
    # Нормализуем ТОЛЬКО строки вида "45450,90689" → "45450.90689"
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

    # 2. Строки → pandas (время отрезаем до парсинга)
    is_str = series.notna() & ~is_num
    if is_str.any():
        str_vals = series[is_str].astype(str).str.strip()

        # 2a. ISO-даты с временем: "2024-09-30 14:15:10" → "2024-09-30"
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

        # 2b. Остальные строки — через pandas с dayfirst
        rest_mask = ~iso_mask
        if rest_mask.any():
            rest_vals = str_vals[rest_mask]
            rest_vals = rest_vals.str.replace(
                r"[ T]\d{1,2}:\d{2}(?::\d{2})?.*$", "", regex=True
            )
            parsed = pd.to_datetime(rest_vals, dayfirst=True, errors="coerce")
            result.loc[rest_vals.index] = parsed.values

    # 3. Остаток → медленный парсер
    unresolved = result.isna() & series.notna()
    if unresolved.any():
        extra = series[unresolved].apply(
            lambda v: parse_visit_date(v, sheet_year=sheet_year)
        )
        result.loc[unresolved] = pd.to_datetime(extra, errors="coerce")

    return pd.to_datetime(result, errors="coerce")


# ---------------------------------------------------------------
# Чтение Excel
# ---------------------------------------------------------------
def _open_excel(path: str) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(path, engine="calamine")
    except Exception:
        return pd.ExcelFile(path, engine="openpyxl")


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                mapping[target] = alias
                break
    return mapping


def _parse_sheet(
    xls: pd.ExcelFile,
    sheet_name: str,
    sheet_year: int,
    sheet_month: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        df = xls.parse(sheet_name)
    except Exception as e:
        st.warning(f"Лист «{sheet_name}» не прочитан: {e}")
        return None, None

    df.columns = [str(c).strip() for c in df.columns]

    mapping = _resolve_columns(df)
    missing = [t for t in COLUMN_ALIASES if t not in mapping]

    # Не пропускаем лист — предупреждаем и заполняем пусто
    if missing:
        lines = [
            f"Лист «{sheet_name}» {TARGET_LABELS[t]} = пусто: нет колонок ['{t}']"
            for t in missing
        ]
        st.warning("\n".join(lines))

    # Собираем итоговый DataFrame
    out = pd.DataFrame(index=df.index)
    for target in COLUMN_ALIASES:
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

    out["Дата визита"] = pd.to_datetime(
        parse_dates_vectorized(out["Дата визита"], sheet_year),
        errors="coerce",
    )
    out["Дата визита"] = out["Дата визита"].dt.normalize()

    # Битые даты запоминаем ДО подстановки — для отчёта
    bad_mask = out["Дата визита"].isna()

    # Подставляем 1-е число месяца вкладки для ВСЕХ NaT,
    # чтобы колонка была заполнена всегда
    fallback_date = pd.Timestamp(year=sheet_year, month=sheet_month, day=1)
    out.loc[bad_mask, "Дата визита"] = fallback_date

    good_rows = out[~bad_mask]
    bad_rows = out[bad_mask]

    # Отчёт только если колонка даты была в исходнике
    bad_report = None
    if "Дата визита" in mapping and not bad_rows.empty:
        bad_report = pd.DataFrame({
            "Вкладка": sheet_name,
            "FinishTime (оригинал)": raw_finish.loc[bad_rows.index].values,
            "CritID": (
                crit_id.loc[bad_rows.index].values
                if crit_id is not None
                else None
            ),
        })

    return good_rows, bad_report


def _parse_sheet_name(sheet_name: str) -> tuple[int, int] | None:
    name = sheet_name.strip()
    if len(name) != 4 or not name.isdigit():
        return None
    month = int(name[:2])
    year = 2000 + int(name[2:])
    if not (1 <= month <= 12):
        return None
    return year, month


# ---------------------------------------------------------------
# Конвертация
# ---------------------------------------------------------------
def convert_excel_to_parquet(
    excel_path: str,
    parquet_path: Path,
    progress_callback=None,
) -> dict:
    xls = _open_excel(excel_path)
    sheet_names = xls.sheet_names

    if not sheet_names:
        raise ValueError("В файле нет вкладок")

    frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    skipped_sheets: list[str] = []
    total = len(sheet_names)

    for i, sheet_name in enumerate(sheet_names, start=1):
        if progress_callback:
            progress_callback(i, total, sheet_name)

        ym = _parse_sheet_name(sheet_name)
        if ym is None:
            skipped_sheets.append(f"{sheet_name} (имя не в формате ММГГ)")
            continue

        year, month = ym
        good, bad = _parse_sheet(xls, sheet_name, year, month)

        if good is None:
            skipped_sheets.append(sheet_name)
            continue

        if not good.empty:
            good = good.drop(
                columns=[c for c in ("Год", "Месяц") if c in good.columns]
            )
            good.insert(0, "Год", year)
            good.insert(1, "Месяц", month)
            frames.append(good)

        if bad is not None and not bad.empty:
            error_frames.append(bad)

        del good, bad
        gc.collect()

    if not frames:
        raise ValueError(
            "Не удалось прочитать ни одной вкладки. "
            f"Пропущены: {skipped_sheets}"
        )

    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    combined = combined[FINAL_COLUMNS]
    
    # Приводим все текстовые колонки к StringDtype:
    # защита от смешанных str/int в исходных данных
    text_cols = [
        c for c in FINAL_COLUMNS
        if c not in ("Год", "Месяц", "Дата визита", "RS ID")
    ]
    for col in text_cols:
        combined[col] = combined[col].astype("string")
    
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

    error_report = pd.concat(error_frames, ignore_index=True) if error_frames else None

    stats = {
        "rows": len(combined),
        "columns": len(combined.columns),
        "sheets_used": total - len(skipped_sheets),
        "skipped": skipped_sheets,
        "parquet_size_mb": parquet_path.stat().st_size / (1024 * 1024),
        "bad_dates": len(error_report) if error_report is not None else 0,
        "error_report": error_report,
    }
    return stats


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Excel → Parquet", layout="wide")
st.title("ЗП Загрузка Excel и конвертация в Parquet")

uploaded = st.file_uploader(
    "Загрузите Excel-файл (вкладки в формате ММГГ)",
    type=["xlsx", "xlsm"],
)

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    st.info(f"Файл сохранён: `{tmp_path}`")

    if st.button("Конвертировать в Parquet", type="primary"):
        progress = st.progress(0.0, text="Читаю вкладки…")

        def _cb(i, total, sheet_name):
            progress.progress(
                i / total,
                text=f"Обрабатываю вкладку {i}/{total}: {sheet_name}",
            )

        try:
            with st.spinner("Конвертирую…"):
                stats = convert_excel_to_parquet(
                    tmp_path, PARQUET_PATH, progress_callback=_cb
                )
            progress.empty()

            st.success("Готово!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Строк", f"{stats['rows']:,}")
            col2.metric("Колонок", stats["columns"])
            col3.metric("Размер Parquet, МБ", f"{stats['parquet_size_mb']:.2f}")

            if stats["skipped"]:
                st.warning("Пропущенные вкладки: " + ", ".join(stats["skipped"]))

            st.download_button(
                "Скачать Parquet",
                data=PARQUET_PATH.read_bytes(),
                file_name="report_data.parquet",
                mime="application/octet-stream",
            )

            if stats["bad_dates"] > 0:
                st.warning(
                    f"Не распознано дат визита: {stats['bad_dates']}. "
                    "Скачайте отчёт ниже."
                )
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    stats["error_report"].to_excel(
                        writer, index=False, sheet_name="BadDates"
                    )
                st.download_button(
                    "Скачать Excel с ошибками дат",
                    data=buffer.getvalue(),
                    file_name="bad_dates.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.subheader("Превью результата (первые 100 строк)")

            preview_df = pd.read_parquet(PARQUET_PATH).head(100)
            st.dataframe(preview_df, use_container_width=True)

            # Выгрузка этих же 100 строк в Excel
            preview_buffer = io.BytesIO()
            with pd.ExcelWriter(preview_buffer, engine="openpyxl") as writer:
                preview_df.to_excel(
                    writer, index=False, sheet_name="Preview100"
                )
            st.download_button(
                "Скачать эти 100 строк в Excel",
                data=preview_buffer.getvalue(),
                file_name="preview_100.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
