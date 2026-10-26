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
    "Регион long",
    "RS",
    "ТП",
    "Дата визита",
    "План ТП, руб",
    "Факт ТП, руб",
    "ПФ ТП, руб",
    "Дата старта",
    "Дата финиша",
    "Длительность",
]

FINAL_COLUMNS = REQUIRED_COLUMNS

CATEGORICAL_COLUMNS = [
    "Тип файла", "Клиент", "Код проекта", "Волна", "ASM", "Регион short",
]

PARQUET_PATH = Path("data/report_data.parquet")
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)


# ===============================================================
# СЛОВАРИ РЕГИОНОВ
# ===============================================================
REGION_MAPPING = {
    'AD': 'Республика Адыгея', 'AL': 'Алтайский край', 'AM': 'Амурская область',
    'AR': 'Архангельская область', 'AS': 'Астраханская область', 'BK': 'Республика Башкортостан',
    'BL': 'Белгородская область', 'BR': 'Брянская область', 'BU': 'Республика Бурятия',
    'CK': 'Чукотский автономный округ', 'CL': 'Челябинская область', 'CN': 'Чеченская Республика',
    'CV': 'Чувашская Республика', 'DA': 'Республика Дагестан', 'DN': 'Донецкая Народная Республика',
    'GA': 'Республика Алтай', 'IN': 'Республика Ингушетия', 'IR': 'Иркутская область',
    'IV': 'Ивановская область', 'KA': 'Камчатский край', 'KB': 'Кабардино-Балкарская Республика',
    'KC': 'Карачаево-Черкесская Республика', 'KD': 'Краснодарский край', 'KE': 'Кемеровская область',
    'KG': 'Калужская область', 'KH': 'Хабаровский край', 'KI': 'Республика Карелия',
    'KK': 'Республика Хакасия', 'KL': 'Республика Калмыкия', 'KM': 'Ханты-Мансийский автономный округ',
    'KN': 'Калининградская область', 'KO': 'Республика Коми', 'KS': 'Курская область',
    'KT': 'Костромская область', 'KU': 'Курганская область', 'KV': 'Кировская область',
    'KY': 'Красноярский край', 'LG': 'Луганская Народная Республика', 'LN': 'Ленинградская область',
    'LP': 'Липецкая область', 'MC': 'Москва', 'ME': 'Республика Марий Эл',
    'MG': 'Магаданская область', 'MM': 'Мурманская область', 'MR': 'Республика Мордовия',
    'MS': 'Московская область', 'NG': 'Новгородская область', 'NN': 'Ненецкий автономный округ',
    'NO': 'Республика Северная Осетия', 'NS': 'Новосибирская область', 'NZ': 'Нижегородская область',
    'OB': 'Оренбургская область', 'OL': 'Орловская область', 'OM': 'Омская область',
    'PE': 'Пермский край', 'PR': 'Приморский край', 'PS': 'Псковская область',
    'PZ': 'Пензенская область', 'RK': 'Республика Крым', 'RO': 'Ростовская область',
    'RZ': 'Рязанская область', 'SA': 'Самарская область', 'SK': 'Республика Саха (Якутия)',
    'SL': 'Сахалинская область', 'SM': 'Смоленская область', 'SR': 'Саратовская область',
    'ST': 'Ставропольский край', 'SV': 'Свердловская область', 'TB': 'Тамбовская область',
    'TL': 'Тульская область', 'TO': 'Томская область', 'TT': 'Республика Татарстан',
    'TU': 'Республика Тыва', 'TV': 'Тверская область', 'TY': 'Тюменская область',
    'UD': 'Удмуртская Республика', 'UL': 'Ульяновская область', 'VG': 'Волгоградская область',
    'VL': 'Владимирская область', 'VO': 'Вологодская область', 'VR': 'Воронежская область',
    'YN': 'Ямало-Ненецкий автономный округ', 'YS': 'Ярославская область', 'YV': 'Еврейская автономная область',
    'ZK': 'Забайкальский край', 'ZO': 'Запорожская область',
    'AA': 'Минская область',
}

long_to_short_region = {
    'Республика Адыгея': 'AD', 'Алтайский край': 'AL', 'Алтай': 'AL', 'Амурская область': 'AM',
    'Архангельская область': 'AR', 'Астраханская область': 'AS', 'Республика Башкортостан': 'BK',
    'Белгородская область': 'BL', 'Брянская область': 'BR', 'Республика Бурятия': 'BU',
    'Чукотский автономный округ': 'CK', 'Челябинская область': 'CL', 'Чеченская Республика': 'CN',
    'Чувашская Республика': 'CV', 'Республика Дагестан': 'DA', 'Донецкая Народная Республика': 'DN',
    'Республика Алтай': 'GA', 'Республика Ингушетия': 'IN', 'Иркутская область': 'IR',
    'Ивановская область': 'IV', 'Камчатский край': 'KA', 'Кабардино-Балкарская Республика': 'KB',
    'Карачаево-Черкесская Республика': 'KC', 'Краснодарский край': 'KD', 'Кемеровская область': 'KE',
    'Калужская область': 'KG', 'Хабаровский край': 'KH', 'Республика Карелия': 'KI',
    'Республика Хакасия': 'KK', 'Республика Калмыкия': 'KL', 'Ханты-Мансийский автономный округ': 'KM',
    'Калининградская область': 'KN', 'Республика Коми': 'KO', 'Курская область': 'KS',
    'Костромская область': 'KT', 'Курганская область': 'KU', 'Кировская область': 'KV',
    'Красноярский край': 'KY', 'Луганская Народная Республика': 'LG', 'Ленинградская область': 'LN',
    'Липецкая область': 'LP', 'Московская область': 'MS', 'Москва': 'MC', 'Республика Марий Эл': 'ME',
    'Магаданская область': 'MG', 'Мурманская область': 'MM', 'Республика Мордовия': 'MR',
    'Новгородская область': 'NG', 'Ненецкий автономный округ': 'NN', 'Республика Северная Осетия': 'NO',
    'Новосибирская область': 'NS', 'Нижегородская область': 'NZ', 'Оренбургская область': 'OB',
    'Орловская область': 'OL', 'Омская область': 'OM', 'Пермский край': 'PE',
    'Приморский край': 'PR', 'Псковская область': 'PS', 'Пензенская область': 'PZ',
    'Республика Крым': 'RK', 'Ростовская область': 'RO', 'Рязанская область': 'RZ',
    'Самарская область': 'SA', 'Республика Саха (Якутия)': 'SK', 'Сахалинская область': 'SL',
    'Смоленская область': 'SM', 'Саратовская область': 'SR', 'Ставропольский край': 'ST',
    'Свердловская область': 'SV', 'Тамбовская область': 'TB', 'Тульская область': 'TL',
    'Томская область': 'TO', 'Республика Татарстан': 'TT', 'Республика Тыва': 'TU',
    'Тверская область': 'TV', 'Тюменская область': 'TY', 'Удмуртская Республика': 'UD',
    'Ульяновская область': 'UL', 'Волгоградская область': 'VG', 'Владимирская область': 'VL',
    'Вологодская область': 'VO', 'Воронежская область': 'VR', 'Ямало-Ненецкий автономный округ': 'YN',
    'Ярославская область': 'YS', 'Еврейская автономная область': 'YV', 'Забайкальский край': 'ZK',
    'Запорожская область': 'ZO', 'Республика Чувашия': 'CV', 'Марий Эл': 'ME', 'Удмуртия': 'UD',
    'Херсонская область': 'KS', 'Хабаровск': 'KH', 'Минская область': 'AA',
}

region_keywords = {
    'AD': ['адыг'], 'AL': ['алтай'], 'AM': ['амур'],
    'AR': ['архангельск'], 'AS': ['астрахан'], 'BK': ['башкортостан', 'башкир'],
    'BL': ['белгород'], 'BR': ['брянск'], 'BU': ['бурят'],
    'CL': ['челябин'], 'CN': ['чечен'], 'CV': ['чуваш'],
    'DA': ['дагестан'], 'DN': ['донецк', 'донецкая'], 'GA': ['горный алтай', 'республика алтай'],
    'IN': ['ингуш'], 'IR': ['иркут'], 'IV': ['иван'],
    'KA': ['камчат'], 'KB': ['кабард'], 'KC': ['карача'],
    'KD': ['краснодар'], 'KE': ['кемер'], 'KG': ['калуж'],
    'KH': ['хабаров'], 'KI': ['карел'], 'KK': ['хакас'],
    'KL': ['калмы'], 'KM': ['хант', 'манс'], 'KN': ['калинин'],
    'KO': ['коми'], 'KS': ['курск'], 'KT': ['костр'],
    'KU': ['курган'], 'KV': ['киров'], 'KY': ['краснояр'],
    'LG': ['луганск', 'луганская'], 'LN': ['ленинград', 'питер'], 'LP': ['липец'],
    'MC': ['моск'], 'ME': ['марий'], 'MG': ['магадан'],
    'MM': ['мурман'], 'MR': ['мордов'], 'MS': ['моск'],
    'NG': ['новгород'], 'NN': ['ненец'], 'NO': ['осет'],
    'NS': ['новосиб'], 'NZ': ['нижегород'], 'OB': ['оренбург'],
    'OL': ['орлов'], 'OM': ['омск'], 'PE': ['перм'],
    'PR': ['примор'], 'PS': ['псков'], 'PZ': ['пенз'],
    'RK': ['крым'], 'RO': ['ростов'], 'RZ': ['ряз'],
    'SA': ['самар'], 'SK': ['саха', 'якут'], 'SL': ['сахалин'],
    'SM': ['смол'], 'SR': ['саратов'], 'ST': ['ставроп'],
    'SV': ['свердлов'], 'TB': ['тамбов'], 'TL': ['туль'],
    'TO': ['томск'], 'TT': ['татар'], 'TU': ['тыва'],
    'TV': ['твер'], 'TY': ['тюмен'], 'UD': ['удмурт'],
    'UL': ['ульян'], 'VG': ['волгоград'], 'VL': ['владимир'],
    'VO': ['волог'], 'VR': ['воронеж'], 'YN': ['ямал'],
    'YS': ['ярослав'], 'YV': ['еврей'], 'ZK': ['забайкал'],
    'ZO': ['запорожье', 'запорожская']
}


def normalize_region(value) -> tuple[str | None, str | None]:
    if pd.isna(value):
        return None, None

    text = str(value).strip().lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)

    if not text:
        return None, None

    # 1. Точное совпадение всей строки с длинным названием
    if text in {k.lower() for k in long_to_short_region}:
        for long_name, code in long_to_short_region.items():
            if long_name.lower() == text:
                return code, REGION_MAPPING.get(code, long_name)

    # 2. Поиск 2-буквенного кода в начале строки или отделенным пробелом/дефисом
    m = re.search(r"(?:^|[\s\-])([a-z]{2})(?:[\s\-]|$)", text)
    if m:
        code = m.group(1).upper()
        if code in REGION_MAPPING:
            return code, REGION_MAPPING[code]

    # 3. Поиск полного длинного названия как подстроки
    for long_name, code in long_to_short_region.items():
        if long_name.lower() in text:
            return code, REGION_MAPPING.get(code, long_name)

    # 4. Поиск по ключевым словам
    for code, keywords in region_keywords.items():
        for kw in keywords:
            if kw in text:
                return code, REGION_MAPPING.get(code)

    return None, None


def validate_parquet(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)


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
    "RS": ["ЭМ", "RS"],
    "ТП": ["Тайный покупатель Fullname", "Респондент"],
    "ВСЕГО": ["ВСЕГО", "Всего", "Итого"],
    "Компенсация": ["Компенсация за покупку", "Компенсация"],
    "Бонус": ["Бонус", "Бонусы"],
}

DEFAULT_HVOSTY_ALIASES = {
    "ASM": DEFAULT_ZP_ALIASES["ASM"],
    "Дата визита": [],
    "Волна": ["Название волны", "SetName"],
    "Код проекта": ["Код проекта"] + DEFAULT_ZP_ALIASES["Код проекта"],
    "Клиент": ["Проект", "Проекты"] + DEFAULT_ZP_ALIASES["Клиент"],
    "Регион short": ["Регион"] + DEFAULT_ZP_ALIASES["Регион short"],
    "RS": [],
    "ТП": [],
    "ВСЕГО": [],
    "Компенсация": [],
    "Бонус": [],
}

TARGET_LABELS = {
    "ASM": "АСМ",
    "Дата визита": "Дата визита",
    "Волна": "Волна",
    "Код проекта": "Код проекта",
    "Клиент": "Клиент",
    "Регион short": "Регион short",
    "RS": "РС",
    "ТП": "ТП",
    "ВСЕГО": "ВСЕГО",
    "Компенсация": "Компенсация",
    "Бонус": "Бонус",
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
        if t not in mapping and aliases[t] and t not in ("ВСЕГО", "Компенсация", "Бонус")
    ]
    if missing:
        lines = [
            f"[{file_type}] Лист «{sheet_name}» "
            f"{TARGET_LABELS.get(t, t)} = пусто: нет колонок ['{t}']"
            for t in missing
        ]
        st.warning("\n".join(lines))

    out = pd.DataFrame(index=df.index)
    for target in aliases:
        if target in mapping:
            out[target] = df[mapping[target]]
        else:
            out[target] = pd.Series([pd.NA] * len(df), index=df.index)

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

    # --- РАСЧЕТ ФАКТ ТП ---
    if is_zp:
        def get_numeric_series(col_name):
            if col_name in out.columns:
                return pd.to_numeric(out[col_name], errors="coerce")
            return pd.Series(pd.NA, index=df.index)

        vsego = get_numeric_series("ВСЕГО")
        komp = get_numeric_series("Компенсация")
        bonus = get_numeric_series("Бонус")
        
        has_data = vsego.notna()
        out["Факт ТП, руб"] = pd.NA
        out.loc[has_data, "Факт ТП, руб"] = (
            vsego[has_data].fillna(0) - 
            komp[has_data].fillna(0) - 
            bonus[has_data].fillna(0)
        )
    else:
        out["Факт ТП, руб"] = pd.NA

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


def _parse_sheet_name_ps(sheet_name: str) -> tuple[int, int] | None:
    name = sheet_name.strip()
    if len(name) != 4 or not name.isdigit():
        return None
    month = int(name[:2])
    year = 2000 + int(name[2:])
    if not (1 <= month <= 12):
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
            
            for c in ["ВСЕГО", "Компенсация", "Бонус"]:
                if c in good.columns:
                    good = good.drop(columns=[c])
                    
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
# БДР
# ===============================================================
def parse_bdr(bdr_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(bdr_bytes))
    except Exception as e:
        st.error(f"Не удалось прочитать БДР: {e}")
        return pd.DataFrame(columns=["Код проекта", "План ТП, руб"])

    df.columns = [str(c).strip() for c in df.columns]
    
    target_col_name = "расходы на тп/респонденты (план на 1 ед.)"
    
    code_col = None
    plan_col = None
    
    for col in df.columns:
        if "КОД" in col.upper():
            code_col = col
        normalized_col = " ".join(col.lower().split())
        if normalized_col == target_col_name:
            plan_col = col
            
    if not code_col or not plan_col:
        st.warning("В файле БДР не найдены колонки 'КОД' или 'Расходы на ТП/Респонденты (план на 1 ед.)'")
        return pd.DataFrame(columns=["Код проекта", "План ТП, руб"])
        
    df = df[[code_col, plan_col]].dropna(subset=[code_col]).copy()
    df.columns = ["Код проекта", "План ТП, руб"]
    
    df["Код проекта"] = df["Код проекта"].astype(str).str.strip()
    df["План ТП, руб"] = pd.to_numeric(df["План ТП, руб"], errors="coerce")
    
    df = df.drop_duplicates(subset=["Код проекта"], keep="first")
    return df


# ===============================================================
# ПС (Проекты Сервизория)
# ===============================================================
def parse_ps(ps_bytes: bytes) -> pd.DataFrame:
    empty = pd.DataFrame(columns=[
        "Год", "Месяц", "Клиент", "Код проекта",
        "Дата старта", "Дата финиша",
    ])
    try:
        xls = pd.ExcelFile(io.BytesIO(ps_bytes), engine="openpyxl")
    except Exception as e:
        st.error(f"Не удалось прочитать файл ПС: {e}")
        return empty

    all_frames = []
    for sheet_name in xls.sheet_names:
        ym = _parse_sheet_name_ps(sheet_name)
        if ym is None:
            continue
        year, month = ym

        try:
            df = xls.parse(sheet_name)
        except Exception:
            continue

        df.columns = [str(c).strip() for c in df.columns]

        col_client = col_code = col_start = col_end = None
        for c in df.columns:
            n = " ".join(c.lower().split())
            if n in ("клиент на по", "проекты в https://ru.checker-soft.com", "проекты в shopmetrics"):
                col_client = c
            elif "код проекта" in n:
                col_code = c
            elif "дата старта" in n:
                col_start = c
            elif "дата финиша" in n:
                col_end = c

        if not col_client or not col_start or not col_end:
            continue

        sub = df[[col_client, col_start, col_end]].copy()
        sub.columns = ["Клиент", "Дата старта", "Дата финиша"]
        sub["Код проекта"] = df[col_code] if col_code else pd.NA

        sub["Клиент"] = (
            sub["Клиент"].astype(str).str.strip().str.lower()
            .str.replace("\u00a0", " ", regex=False)
        )
        sub = sub[~sub["Клиент"].isin(["nan", "<na>", "none", ""])]
        if sub.empty:
            continue

        sub["Дата старта"] = pd.to_datetime(
            sub["Дата старта"].apply(lambda v: parse_visit_date(v, sheet_year=year)),
            errors="coerce",
        )
        sub["Дата финиша"] = pd.to_datetime(
            sub["Дата финиша"].apply(lambda v: parse_visit_date(v, sheet_year=year)),
            errors="coerce",
        )

        first_day = pd.Timestamp(year=year, month=month, day=1)
        last_day = first_day + pd.offsets.MonthEnd(0)

        sub["Дата старта"] = sub["Дата старта"].fillna(first_day)
        sub["Дата финиша"] = sub["Дата финиша"].fillna(last_day)

        sub.loc[sub["Дата старта"] < first_day, "Дата старта"] = first_day
        sub.loc[sub["Дата финиша"] > last_day, "Дата финиша"] = last_day

        sub["Год"] = year
        sub["Месяц"] = month
        all_frames.append(sub)

    if not all_frames:
        return empty

    combined = pd.concat(all_frames, ignore_index=True)

    def join_codes(s):
        codes = sorted(set(
            str(v).strip() for v in s.dropna()
            if str(v).strip() not in ("", "nan", "<NA>", "None")
        ))
        return ", ".join(codes)

    grouped = combined.groupby(["Год", "Месяц", "Клиент"], as_index=False).agg({
        "Код проекта": join_codes,
        "Дата старта": "min",
        "Дата финиша": "max",
    })

    return grouped


# ===============================================================
# КОНВЕРТАЦИЯ
# ===============================================================
def convert_to_parquet(
    zp_path: str | None,
    hvosty_path: str | None,
    bdr_bytes: bytes | None,
    ps_bytes: bytes | None,
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

    # ===========================================================
    # НОРМАЛИЗАЦИЯ РЕГИОНОВ
    # ===========================================================
    raw_regions = combined["Регион short"].astype(str)
    results = [normalize_region(v) for v in raw_regions]
    combined["_region_short_norm"] = [r[0] for r in results]
    combined["_region_long_norm"] = [r[1] for r in results]

    mask_unrecognized = combined["_region_short_norm"].isna()
    if mask_unrecognized.any():
        bad_regions = combined.loc[mask_unrecognized, ["Тип файла", "Регион short"]].copy()
        bad_regions = bad_regions.drop_duplicates(subset=["Регион short"])
        bad_regions.rename(columns={"Регион short": "Оригинал"}, inplace=True)
    else:
        bad_regions = None

    combined["Регион long"] = combined["_region_long_norm"]
    mask_found = combined["_region_short_norm"].notna()
    combined.loc[mask_found, "Регион short"] = combined.loc[mask_found, "_region_short_norm"]

    # НОВОЕ: для нераспознанных — AA / Не определен
    mask_unrecognized = combined["_region_short_norm"].isna()
    combined.loc[mask_unrecognized, "Регион short"] = "AA"
    combined.loc[mask_unrecognized, "Регион long"] = "Не определен"

    combined = combined.drop(columns=["_region_short_norm", "_region_long_norm"])

    # ===========================================================
    # ОБОГАЩЕНИЕ БДР
    # ===========================================================
    bdr_df = pd.DataFrame()
    if bdr_bytes:
        bdr_df = parse_bdr(bdr_bytes)

    if "План ТП, руб" not in combined.columns:
        combined["План ТП, руб"] = pd.NA

    if not bdr_df.empty and "План ТП, руб" in bdr_df.columns:
        combined["Код проекта"] = (
            combined["Код проекта"]
            .astype(str)
            .str.strip()
            .replace(["nan", "<NA>", "None"], pd.NA)
        )
        if "План ТП, руб" in combined.columns:
            combined = combined.drop(columns=["План ТП, руб"])
        combined = combined.merge(bdr_df, on="Код проекта", how="left")

    combined["План ТП, руб"] = pd.to_numeric(combined["План ТП, руб"], errors="coerce")
    combined["Факт ТП, руб"] = pd.to_numeric(combined["Факт ТП, руб"], errors="coerce")
    
    mask_missing = combined["План ТП, руб"].isna() | combined["Факт ТП, руб"].isna()
    combined.loc[mask_missing, "План ТП, руб"] = pd.NA
    combined.loc[mask_missing, "Факт ТП, руб"] = pd.NA
    
    combined["ПФ ТП, руб"] = pd.NA
    mask_both_filled = combined["План ТП, руб"].notna() & combined["Факт ТП, руб"].notna()
    combined.loc[mask_both_filled, "ПФ ТП, руб"] = (
        combined.loc[mask_both_filled, "План ТП, руб"]
        - combined.loc[mask_both_filled, "Факт ТП, руб"]
    )

    # ===========================================================
    # ОБОГАЩЕНИЕ ПС
    # ===========================================================
    combined["Дата старта"] = pd.to_datetime(
        dict(year=combined["Год"].astype(int), month=combined["Месяц"].astype(int), day=1)
    )
    combined["Дата финиша"] = (
        combined["Дата старта"] + pd.offsets.MonthEnd(0)
    )

    ps_df = pd.DataFrame()
    if ps_bytes:
        ps_df = parse_ps(ps_bytes)

    if not ps_df.empty:
        combined["_клиент_norm"] = (
            combined["Клиент"].astype(str).str.strip().str.lower()
            .str.replace("\u00a0", " ", regex=False)
        )

        ps_client = ps_df[["Год", "Месяц", "Клиент", "Дата старта", "Дата финиша"]].rename(
            columns={"Дата старта": "Дата старта_ps", "Дата финиша": "Дата финиша_ps"}
        )
        
        combined["_old_idx"] = combined.index
        
        merged1 = combined.merge(
            ps_client,
            left_on=["Год", "Месяц", "_клиент_norm"],
            right_on=["Год", "Месяц", "Клиент"],
            how="left",
            suffixes=("", "_right"),
        )
        
        mask_found1 = merged1["Дата старта_ps"].notna()
        
        if mask_found1.any():
            found_idx = merged1.loc[mask_found1, "_old_idx"].values
            combined.loc[found_idx, "Дата старта"] = merged1.loc[mask_found1, "Дата старта_ps"].values
            combined.loc[found_idx, "Дата финиша"] = merged1.loc[mask_found1, "Дата финиша_ps"].values

        mask_not_found = ~mask_found1
        if mask_not_found.any():
            sub = merged1.loc[mask_not_found].copy()
            
            ps_long = ps_df[["Год", "Месяц", "Код проекта", "Дата старта", "Дата финиша"]].copy()
            ps_long["Код проекта"] = ps_long["Код проекта"].astype(str).str.split(", ")
            ps_long = ps_long.explode("Код проекта")
            ps_long = ps_long[ps_long["Код проекта"].notna() & (ps_long["Код проекта"] != "")]
            ps_long = ps_long.drop_duplicates(subset=["Год", "Месяц", "Код проекта"], keep="first")
            
            ps_long = ps_long.rename(
                columns={"Дата старта": "Дата старта_ps2", "Дата финиша": "Дата финиша_ps2"}
            )
            
            sub["Код проекта"] = sub["Код проекта"].astype(str).str.strip()
            
            merged2 = sub.merge(
                ps_long, on=["Год", "Месяц", "Код проекта"], how="left"
            )
            
            mask_found2 = merged2["Дата старта_ps2"].notna()
            if mask_found2.any():
                found2_idx = merged2.loc[mask_found2, "_old_idx"].values
                combined.loc[found2_idx, "Дата старта"] = merged2.loc[mask_found2, "Дата старта_ps2"].values
                combined.loc[found2_idx, "Дата финиша"] = merged2.loc[mask_found2, "Дата финиша_ps2"].values

        combined = combined.drop(columns=["_клиент_norm", "_old_idx"])

    combined["Длительность"] = (
        combined["Дата финиша"] - combined["Дата старта"]
    ).dt.days.astype("Int64")

    combined = combined[FINAL_COLUMNS]

    text_cols = [
        c for c in FINAL_COLUMNS
        if c not in ("Год", "Месяц", "Дата визита", "План ТП, руб", "Факт ТП, руб", "ПФ ТП, руб", "Дата старта", "Дата финиша", "Длительность")
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
        "bad_regions": bad_regions,
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
        "bdr_bytes": None,
        "ps_bytes": None,
        "stats": None,
        "parquet_bytes": None,
        "bad_dates_bytes": None,
        "bad_regions_bytes": None,
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
                st.info("Пожалуйста, пересоберите Parquet на вкладке «Настройки проекта».")
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

    def _month_label(year: int, month: int) -> str:
        return f"{month}_{year % 100:02d}"

    def compute_kpi(
        df_client_all: pd.DataFrame,
        months: list[tuple[int, int]],
    ) -> dict:
        plans = []
        facts = []
        rs_counts = []
        plan_tps = []
        fact_tps = []

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
            
            zp_sub = sub[sub["Тип файла"] == "ЗП"]
            if not zp_sub.empty:
                plan_tp_val = zp_sub["План ТП, руб"].sum(min_count=1)
                fact_tp_val = zp_sub["Факт ТП, руб"].sum(min_count=1)
                if pd.notna(plan_tp_val):
                    plan_tps.append(plan_tp_val)
                if pd.notna(fact_tp_val):
                    fact_tps.append(fact_tp_val)

        n = len(plans)
        if n == 0:
            return {
                "plan_avg": 0, "fact_avg": 0, "pf_percent": 0,
                "rs_count_avg": 0, "ankets_per_rs": 0,
                "plan_tp_avg": 0, "fact_tp_avg": 0, "pf_tp_avg": 0,
            }

        plan_avg = round(sum(plans) / n)
        fact_avg = round(sum(facts) / n)
        rs_avg = round(sum(rs_counts) / n)
        pf_percent = (round(fact_avg / plan_avg * 100) if plan_avg > 0 else 0)
        ankets = (round(fact_avg / rs_avg) if rs_avg > 0 else 0)
        
        plan_tp_avg = round(sum(plan_tps) / len(plan_tps)) if plan_tps else 0
        fact_tp_avg = round(sum(fact_tps) / len(fact_tps)) if fact_tps else 0
        pf_tp_avg = plan_tp_avg - fact_tp_avg

        return {
            "plan_avg": plan_avg, "fact_avg": fact_avg, "pf_percent": pf_percent,
            "rs_count_avg": rs_avg, "ankets_per_rs": ankets,
            "plan_tp_avg": plan_tp_avg, "fact_tp_avg": fact_tp_avg, "pf_tp_avg": pf_tp_avg,
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
            "План ТП, руб": [],
            "Факт ТП, руб": [],
            "ПФ ТП, руб": [],
        }
        for y, m in reversed(months):
            sub = df_client_all[
                (df_client_all["Год"] == y)
                & (df_client_all["Месяц"] == m)
            ]
            if sub.empty:
                for k in rows.keys():
                    rows[k].append(None)
            else:
                plan = len(sub)
                fact = int((sub["Тип файла"] == "ЗП").sum())
                pf = (fact / plan * 100) if plan > 0 else 0.0
                rs_unique = sub["RS"].dropna().nunique()
                ankets = (fact / rs_unique) if rs_unique > 0 else 0.0
                
                zp_sub = sub[sub["Тип файла"] == "ЗП"]
                if not zp_sub.empty:
                    plan_tp = zp_sub["План ТП, руб"].sum(min_count=1)
                    fact_tp = zp_sub["Факт ТП, руб"].sum(min_count=1)
                    pf_tp = plan_tp - fact_tp if pd.notna(plan_tp) and pd.notna(fact_tp) else None
                else:
                    plan_tp = None
                    fact_tp = None
                    pf_tp = None
                
                rows["План проекта, шт"].append(plan)
                rows["Факт проекта, шт"].append(fact)
                rows["ПФ клиент, %"].append(round(pf))
                rows["Кол-во RS, чел."].append(rs_unique)
                rows["Кол-во анкет на 1 RS, шт."].append(round(ankets))
                rows["План ТП, руб"].append(plan_tp)
                rows["Факт ТП, руб"].append(fact_tp)
                rows["ПФ ТП, руб"].append(pf_tp)

        columns = [_month_label(y, m) for y, m in reversed(months)]
        table = pd.DataFrame(rows, index=columns).T
        table.index.name = "Метрика"
        return table

    with rep_clients:
        df_dataset = st.session_state.dataset.get("df")

        if df_dataset is None:
            st.info("Сначала загрузите Parquet на вкладке «Загрузка данных».")
        else:
            st.markdown("### 📥 Выгрузка датасета")
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                pairs = sorted(set(
                    (int(y), int(m)) for y, m in df_dataset[["Год", "Месяц"]]
                    .dropna().itertuples(index=False, name=None)
                ))
                
                period_opts = []
                if pairs:
                    y_min, m_min = pairs[0]
                    y_max, m_max = pairs[-1]
                    
                    y, m = y_min, m_min
                    while (y, m) <= (y_max, m_max):
                        period_opts.append(f"{m:02d}.{y}")
                        if m == 12:
                            y, m = y + 1, 1
                        else:
                            m += 1
                
                period_selected = st.multiselect(
                    "Выберите месяцы для выгрузки",
                    options=period_opts,
                    default=period_opts[-3:] if len(period_opts) >= 3 else period_opts,
                    key="export_period"
                )
            
            with col_f2:
                all_clients = sorted(df_dataset["Клиент"].dropna().unique().astype(str).tolist())
                clients_selected = st.multiselect(
                    "Выберите клиентов для выгрузки",
                    options=all_clients,
                    default=[],
                    key="export_clients"
                )

            if st.button("📥 Сформировать Excel для выгрузки", type="primary"):
                df_export = df_dataset.copy()
                
                if period_selected:
                    parsed_periods = set()
                    for p in period_selected:
                        mm, yyyy = p.split(".")
                        parsed_periods.add(int(yyyy) * 100 + int(mm))
                    
                    ym = (
                        df_export["Год"].astype("int64") * 100
                        + df_export["Месяц"].astype("int64")
                    )
                    df_export = df_export[ym.isin(parsed_periods)]
                    
                if clients_selected:
                    df_export = df_export[df_export["Клиент"].isin(clients_selected)]
                    
                if df_export.empty:
                    st.warning("По выбранным фильтрам данных нет.")
                else:
                    export_buffer = io.BytesIO()
                    with pd.ExcelWriter(export_buffer, engine="openpyxl") as writer:
                        df_export.to_excel(writer, index=False, sheet_name="Данные")
                    
                    st.session_state.export_bytes = export_buffer.getvalue()
                    st.session_state.export_rows = len(df_export)
            
            if st.session_state.get("export_bytes"):
                st.success(f"Файл готов ({st.session_state.export_rows} строк)")
                st.download_button(
                    "Скачать Excel",
                    data=st.session_state.export_bytes,
                    file_name="dataset_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_export_btn"
                )
            st.markdown("---")

            clients = sorted(
                df_dataset["Клиент"].dropna().unique().astype(str).tolist()
            )

            if not clients:
                st.warning("В датасете нет клиентов.")
            else:
                client = st.selectbox(
                    "Клиент", options=clients, key="rep_clients_client"
                )

                df_client_all = df_dataset[df_dataset["Клиент"] == client]

                st.markdown("### 📊 Ключевые показатели")
                report_key = (client,)

                if st.button("Рассчитать", type="primary"):
                    if df_client_all.empty:
                        st.session_state.report = {
                            "key": report_key, "empty": True, "kpi": None, "table": None,
                        }
                    else:
                        pairs = sorted(set(
                            (int(y), int(m)) for y, m in df_client_all[["Год", "Месяц"]]
                            .dropna().itertuples(index=False, name=None)
                        ))
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
                        table = compute_monthly_table(df_client_all, months)
                        st.session_state.report = {
                            "key": report_key, "empty": False, "kpi": kpi, "table": table,
                        }

                report = st.session_state.report
                if report is None or report["key"] != report_key:
                    st.info("Нажмите «Рассчитать», чтобы построить отчёт.")
                elif report["empty"]:
                    st.info("Нет данных по выбранному клиенту.")
                else:
                    kpi = report["kpi"]
                    table = report["table"]

                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("📊 План ср., шт", f"{kpi['plan_avg']:,}")
                    k2.metric("✅ Факт ср., шт", f"{kpi['fact_avg']:,}")
                    k3.metric("🎯 ПФ ср., %", f"{kpi['pf_percent']}%")
                    k4.metric("👥 Кол-во RS ср., чел.", f"{kpi['rs_count_avg']:,}")
                    k5.metric("📋 Кол-во анкет на 1 RS ср., шт.", f"{kpi['ankets_per_rs']}")
                    
                    tk1, tk2, tk3 = st.columns(3)
                    tk1.metric("💰 План ТП ср., руб", f"{kpi['plan_tp_avg']:,.0f}")
                    tk2.metric("💵 Факт ТП ср., руб", f"{kpi['fact_tp_avg']:,.0f}")
                    tk3.metric("📉 ПФ ТП ср., руб", f"{kpi['pf_tp_avg']:,.0f}")

                    st.markdown("### 📈 Ключевые показатели по месяцам")
                    st.dataframe(table, use_container_width=True)

                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        table.to_excel(writer, sheet_name="Ключевые показатели")
                    st.download_button(
                        "📥 Выгрузить в Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"kpi_{client}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                    dataset_buffer = io.BytesIO()
                    with pd.ExcelWriter(dataset_buffer, engine="openpyxl") as writer:
                        df_client_all.to_excel(writer, index=False, sheet_name="Датасет")
                    st.download_button(
                        "📥 Выгрузить датасет (все строки клиента)",
                        data=dataset_buffer.getvalue(),
                        file_name=f"dataset_{client}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                    st.markdown("### 📊 Дашборд")
                    st.markdown("<br>", unsafe_allow_html=True)

                    font = (
                        "Inter, -apple-system, BlinkMacSystemFont, "
                        "'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
                    )

                    def render_bar_chart(
                        df: pd.DataFrame, metric_name: str, title: str, y_title: str,
                        color_rule, is_currency=False
                    ):
                        series = df.loc[metric_name]
                        chart_data = pd.DataFrame({
                            "Месяц": series.index, "Value": series.values
                        }).dropna()

                        if not chart_data.empty:
                            chart_data["Value"] = chart_data["Value"].astype(float)
                            chart_data = chart_data.iloc[::-1].reset_index(drop=True)
                            chart_data["Цвет"] = chart_data["Value"].apply(color_rule)

                            max_val = chart_data["Value"].max()
                            min_val = chart_data["Value"].min()
                            
                            y_upper = max(max_val * 1.2, 1)
                            y_lower = min(min_val * 1.2, 0) if min_val < 0 else 0

                            bars = alt.Chart(chart_data).mark_bar(
                                cornerRadiusTopLeft=2, cornerRadiusTopRight=2,
                            ).encode(
                                x=alt.X(
                                    "Месяц:N", sort=None,
                                    axis=alt.Axis(labelAngle=0, labelFont=font, title=None),
                                ),
                                y=alt.Y(
                                    "Value:Q",
                                    scale=alt.Scale(domain=[y_lower, y_upper]),
                                    axis=alt.Axis(format=".0f", title=y_title, labelFont=font, titleFont=font),
                                ),
                                color=alt.Color("Цвет:N", scale=None, legend=None),
                            )

                            text = bars.mark_text(
                                align="center", baseline="bottom", dy=-5,
                                font=font, fontSize=12, color="#333333",
                            ).encode(text=alt.Text("Value:Q", format=".0f" if not is_currency else ",.0f"))

                            chart = (bars + text).properties(
                                title=alt.TitleParams(text=title, font=font, fontSize=16, anchor="start"),
                                height=400,
                            ).configure_view(strokeWidth=0).configure_axis(grid=False)

                            st.altair_chart(chart, use_container_width=True)

                    def get_quantile_colors(series, top_is_violet=True):
                        clean_series = series.dropna()
                        if clean_series.empty or clean_series.min() == clean_series.max():
                            return lambda x: "#757575"
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

                    def pf_color(val):
                        if val == 100: return "#6A1B9A"
                        elif val < 80: return "#F50057"
                        else: return "#757575"

                    render_bar_chart(table, "ПФ клиент, %", "ПФ клиент, %", "ПФ клиент, %", pf_color)

                    rs_series = table.loc["Кол-во RS, чел."]
                    render_bar_chart(table, "Кол-во RS, чел.", "Кол-во RS, чел.", "Кол-во RS, чел.", get_quantile_colors(rs_series, True))

                    ankets_series = table.loc["Кол-во анкет на 1 RS, шт."]
                    render_bar_chart(table, "Кол-во анкет на 1 RS, шт.", "Кол-во анкет на 1 RS, шт.", "Кол-во анкет на 1 RS, шт.", get_quantile_colors(ankets_series, False))
                    
                    pf_tp_series = table.loc["ПФ ТП, руб"]
                    render_bar_chart(table, "ПФ ТП, руб", "ПФ ТП, руб", "ПФ ТП, руб", get_quantile_colors(pf_tp_series, True), is_currency=True)

    with rep_regions: st.info("В разработке.")
    with rep_dsm: st.info("В разработке.")
    with rep_summary: st.info("В разработке.")


# ---------------------------------------------------------------
# ВКЛАДКА «НАСТРОЙКИ ПРОЕКТА»
# ---------------------------------------------------------------
with tab_settings:
    st.header("Настройки проекта")
    st.caption(
        "Конвертация Excel (ЗП + Хвосты) и обогащение БДР, ПС → Parquet. "
        "Выполняется раз в месяц, результат сохраняется локально."
    )

    zp_file = st.file_uploader("Файл ЗП (вкладки в формате ММГГ, например 0125)", type=["xlsx", "xlsm"], key="settings_zp")
    hvosty_file = st.file_uploader("Файл Хвосты (вкладки в формате 'Хвосты 09.2026')", type=["xlsx", "xlsm"], key="settings_hvosty")
    bdr_file = st.file_uploader("Файл БДР (Excel, для колонок План ТП, Факт ТП, ПФ ТП)", type=["xlsx", "xlsm"], key="settings_bdr")
    ps_file = st.file_uploader("Файл Проекты Сервизория (Excel, для колонок Дата старта, Дата финиша)", type=["xlsx", "xlsm"], key="settings_ps")

    if zp_file is not None: st.session_state.convert["zp_bytes"] = zp_file.getvalue()
    if hvosty_file is not None: st.session_state.convert["hvosty_bytes"] = hvosty_file.getvalue()
    if bdr_file is not None: st.session_state.convert["bdr_bytes"] = bdr_file.getvalue()
    if ps_file is not None: st.session_state.convert["ps_bytes"] = ps_file.getvalue()

    if st.button("Сформировать датасет ПС", type="secondary"):
        ps_bytes = st.session_state.convert.get("ps_bytes")
        if ps_bytes is None:
            st.error("Загрузите файл ПС.")
        else:
            ps_df = parse_ps(ps_bytes)
            if ps_df.empty:
                st.warning("ПС пуст — нет валидных вкладок.")
            else:
                ps_buffer = io.BytesIO()
                with pd.ExcelWriter(ps_buffer, engine="openpyxl") as writer:
                    ps_df.to_excel(writer, index=False, sheet_name="ПС")
                st.success(f"ПС собран: {len(ps_df):,} строк")
                st.download_button(
                    "📥 Скачать ПС в Excel",
                    data=ps_buffer.getvalue(),
                    file_name="ps_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

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
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                        tmp.write(conv["zp_bytes"])
                        zp_path = tmp.name
                        tmp_paths.append(zp_path)

                if conv["hvosty_bytes"] is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                        tmp.write(conv["hvosty_bytes"])
                        hvosty_path = tmp.name
                        tmp_paths.append(hvosty_path)

                progress = st.progress(0.0, text="Читаю вкладки…")

                def _cb(i, total, sheet_name, file_type):
                    progress.progress(min(i / total, 1.0), text=f"[{file_type}] {i}/{total}: {sheet_name}")

                try:
                    with st.spinner("Конвертирую…"):
                        stats = convert_to_parquet(
                            zp_path, hvosty_path, conv["bdr_bytes"], conv["ps_bytes"], PARQUET_PATH,
                            st.session_state.aliases["zp"], st.session_state.aliases["hvosty"],
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
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        stats["error_report"].to_excel(writer, index=False, sheet_name="BadDates")
                    conv["bad_dates_bytes"] = buffer.getvalue()
                else:
                    conv["bad_dates_bytes"] = None

                if stats.get("bad_regions") is not None and not stats["bad_regions"].empty:
                    reg_buffer = io.BytesIO()
                    with pd.ExcelWriter(reg_buffer, engine="openpyxl") as writer:
                        stats["bad_regions"].to_excel(writer, index=False, sheet_name="BadRegions")
                    conv["bad_regions_bytes"] = reg_buffer.getvalue()
                else:
                    conv["bad_regions_bytes"] = None

                if stats["hvosty_included_names"] or stats["hvosty_excluded_names"]:
                    sheets_buffer = io.BytesIO()
                    with pd.ExcelWriter(sheets_buffer, engine="openpyxl") as writer:
                        if stats["hvosty_included_names"]:
                            pd.DataFrame({"Хвосты добавлены": stats["hvosty_included_names"]}).to_excel(writer, index=False, sheet_name="Included")
                        if stats["hvosty_excluded_names"]:
                            pd.DataFrame({"Вкладки Хвост исключены": stats["hvosty_excluded_names"]}).to_excel(writer, index=False, sheet_name="Excluded")
                    conv["sheets_bytes"] = sheets_buffer.getvalue()
                else:
                    conv["sheets_bytes"] = None

            finally:
                for p in tmp_paths:
                    try: os.unlink(p)
                    except OSError: pass

    conv = st.session_state.convert
    if conv["stats"] is not None:
        stats = conv["stats"]

        st.success("Готово!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Строк", f"{stats['rows']:,}")
        col2.metric("Вкладок обработано", f"{stats['sheets_used']} / {stats['sheets_total']}")
        col3.metric("Размер Parquet, МБ", f"{stats['parquet_size_mb']:.2f}")

        if conv["parquet_bytes"]:
            st.download_button("Скачать Parquet", data=conv["parquet_bytes"], file_name="report_data.parquet", mime="application/octet-stream")

        if conv["bad_dates_bytes"]:
            st.warning(f"Не распознано дат визита: {stats['bad_dates']}. Скачайте отчёт ниже.")
            st.download_button("Скачать Excel с ошибками дат", data=conv["bad_dates_bytes"], file_name="bad_dates.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if conv.get("bad_regions_bytes"):
            st.warning(f"Не распознано регионов: {len(stats['bad_regions'])}. Скачайте отчёт ниже.")
            st.download_button("Скачать Excel с ошибками регионов", data=conv["bad_regions_bytes"], file_name="bad_regions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if conv["sheets_bytes"]:
            st.download_button("Скачать Excel со списком вкладок Хвостов (Included/Excluded)", data=conv["sheets_bytes"], file_name="hvosty_sheets.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
