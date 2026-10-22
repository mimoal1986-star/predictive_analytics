import gc
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Константы
# ---------------------------------------------------------------
COLUMN_RENAME = {
    "ACC": "ASM",
    "FinishTime": "Дата визита",
    "SetName": "Волна",
    "SetCode": "Код проекта",
    "ClientName": "Клиент",
    "RegionName": "Регион short",
    "Тайный покупатель Fullname": "RS",
    "Тайный покупатель ID": "RS ID",
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
# Чтение Excel
# ---------------------------------------------------------------
def _open_excel(path: str) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(path, engine="calamine")
    except Exception:
        return pd.ExcelFile(path, engine="openpyxl")


def _parse_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame | None:
    try:
        df = xls.parse(sheet_name)
    except Exception as e:
        st.warning(f"Лист «{sheet_name}» не прочитан: {e}")
        return None

    df.columns = [str(c).strip() for c in df.columns]

    # FIX 1.3: сообщаем о дубликатах нужных колонок (pandas делает ACC.1)
    dupes = [c for c in COLUMN_RENAME if f"{c}.1" in df.columns]
    if dupes:
        st.warning(
            f"Лист «{sheet_name}»: дубликаты колонок {dupes} — взята первая"
        )

    missing = [c for c in COLUMN_RENAME if c not in df.columns]
    if missing:
        st.warning(f"Лист «{sheet_name}» пропущен: нет колонок {missing}")
        return None

    return df[list(COLUMN_RENAME.keys())].rename(columns=COLUMN_RENAME)


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

    # FIX 1.6: защита от пустого Excel
    if not sheet_names:
        raise ValueError("В файле нет вкладок")

    frames: list[pd.DataFrame] = []
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
        df = _parse_sheet(xls, sheet_name)
        if df is None:
            skipped_sheets.append(sheet_name)
            continue

        # FIX 1.2: не падаем, если Год/Месяц уже есть в исходных данных
        df = df.drop(columns=[c for c in ("Год", "Месяц") if c in df.columns])
        df.insert(0, "Год", year)
        df.insert(1, "Месяц", month)

        frames.append(df)
        del df
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

    combined["Год"] = combined["Год"].astype("int16")
    combined["Месяц"] = combined["Месяц"].astype("int8")

    for col in CATEGORICAL_COLUMNS:
        combined[col] = combined[col].astype("category")

    # FIX 1.4: сообщаем о нераспознанных датах
    dates = pd.to_datetime(
        combined["Дата визита"], dayfirst=True, errors="coerce"
    )
    bad_dates = dates.isna() & combined["Дата визита"].notna()
    if bad_dates.any():
        st.warning(f"Дата визита: {bad_dates.sum()} значений не распознаны")
    combined["Дата визита"] = dates

    # FIX 1.1: безопасное приведение RS ID к Int64
    rs_id = pd.to_numeric(combined["RS ID"], errors="coerce")
    bad_rs = rs_id.notna() & (rs_id % 1 != 0)
    if bad_rs.any():
        st.warning(f"RS ID: {bad_rs.sum()} дробных значений — округлены")
    combined["RS ID"] = rs_id.round().astype("Int64")

    combined.to_parquet(parquet_path, index=False, compression="zstd")

    stats = {
        "rows": len(combined),
        "columns": len(combined.columns),
        "sheets_used": total - len(skipped_sheets),
        "skipped": skipped_sheets,
        "parquet_size_mb": parquet_path.stat().st_size / (1024 * 1024),
    }
    return stats


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Excel → Parquet", layout="wide")
st.title("Загрузка Excel и конвертация в Parquet")

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
            col3.metric(
                "Размер Parquet, МБ", f"{stats['parquet_size_mb']:.2f}"
            )

            if stats["skipped"]:
                st.warning(
                    "Пропущенные вкладки: " + ", ".join(stats["skipped"])
                )

            # FIX 1.5: читаем байты, не держим открытый файл
            st.download_button(
                "Скачать Parquet",
                data=PARQUET_PATH.read_bytes(),
                file_name="report_data.parquet",
                mime="application/octet-stream",
            )

            st.subheader("Превью результата")
            st.dataframe(
                pd.read_parquet(PARQUET_PATH).head(20),
                use_container_width=True,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
