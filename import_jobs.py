from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from data_ingestion import (
    FolderLoadReport,
    load_data_from_folder_with_report,
    load_data_from_uploaded_files,
    persist_core_cost_records,
)


PersistFunc = Callable[[pd.DataFrame, str], int]


@dataclass
class ImportJobResult:
    success: bool
    message: str
    dataframe: Optional[pd.DataFrame] = None
    price_col: str = ""
    scanned_files: list[str] = field(default_factory=list)
    loaded_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    synced_rows: int = 0

    @property
    def scanned_file_count(self) -> int:
        return len(self.scanned_files)

    @property
    def loaded_file_count(self) -> int:
        return len(self.loaded_files)

    @property
    def failed_file_count(self) -> int:
        return len(self.failed_files)


class ImportJob:
    """Scan, validate, then write a complete import batch."""

    def __init__(self, persist_func: PersistFunc | None = None, *, require_all_files_valid: bool = True) -> None:
        self.persist_func = persist_func or (lambda df, price_col: persist_core_cost_records(df, price_col, mode="full"))
        self.require_all_files_valid = require_all_files_valid

    def import_folder(self, folder_path: str) -> ImportJobResult:
        report = load_data_from_folder_with_report(folder_path)
        if report.error_message:
            return self._failed(report, report.error_message)

        if report.failed_files and self.require_all_files_valid:
            failed_preview = "；".join(report.failed_files[:5])
            if len(report.failed_files) > 5:
                failed_preview += f"；另有 {len(report.failed_files) - 5} 个文件"
            return self._failed(
                report,
                f"部分文件未导入，已阻止本次写库：{failed_preview}",
            )

        if report.dataframe is None or report.dataframe.empty:
            return self._failed(report, "没有可写入的有效数据")

        synced_rows = self.persist_func(report.dataframe, report.price_col or "")
        expected_rows = len(report.dataframe)
        if int(synced_rows) != int(expected_rows):
            return ImportJobResult(
                success=False,
                message=f"本地数据同步校验未通过：有效源数据 {expected_rows} 条，实际写入 {synced_rows} 条。",
                dataframe=report.dataframe,
                price_col=report.price_col or "",
                scanned_files=report.scanned_files,
                loaded_files=report.loaded_files,
                failed_files=report.failed_files,
                synced_rows=int(synced_rows or 0),
            )

        return ImportJobResult(
            success=True,
            message=(
                f"已扫描 {len(report.scanned_files)} 个文件，成功导入 {len(report.loaded_files)} 个文件，"
                f"共 {expected_rows} 条记录，并写入本地数据库 {synced_rows} 条记录。"
            ),
            dataframe=report.dataframe,
            price_col=report.price_col or "",
            scanned_files=report.scanned_files,
            loaded_files=report.loaded_files,
            failed_files=report.failed_files,
            synced_rows=int(synced_rows or 0),
        )

    def import_uploaded(self, uploaded_files: list[object]) -> ImportJobResult:
        scanned_files = [str(getattr(file, "name", "未知文件") or "未知文件") for file in uploaded_files or []]
        load_result = load_data_from_uploaded_files(uploaded_files)
        dataframe, price_col, error_message, failed_files = load_result
        failed_names = {str(failure).split(":", 1)[0].strip() for failure in failed_files}
        loaded_files = [file_name for file_name in scanned_files if file_name not in failed_names]
        report = FolderLoadReport(
            dataframe=dataframe,
            price_col=price_col,
            error_message=error_message,
            scanned_files=scanned_files,
            loaded_files=loaded_files,
            failed_files=failed_files,
        )
        if error_message:
            return self._failed(report, error_message)

        if failed_files and self.require_all_files_valid:
            failed_preview = "；".join(failed_files[:5])
            if len(failed_files) > 5:
                failed_preview += f"；另有 {len(failed_files) - 5} 个文件"
            return self._failed(
                report,
                f"部分上传文件未导入，已阻止本次写库：{failed_preview}",
            )

        if dataframe is None or dataframe.empty:
            return self._failed(report, "没有可写入的有效上传数据")

        synced_rows = self.persist_func(dataframe, price_col or "")
        expected_rows = len(dataframe)
        if int(synced_rows) != int(expected_rows):
            return ImportJobResult(
                success=False,
                message=f"本地数据同步校验未通过：有效源数据 {expected_rows} 条，实际写入 {synced_rows} 条。",
                dataframe=dataframe,
                price_col=price_col or "",
                scanned_files=scanned_files,
                loaded_files=loaded_files,
                failed_files=failed_files,
                synced_rows=int(synced_rows or 0),
            )

        return ImportJobResult(
            success=True,
            message=(
                f"已扫描 {len(scanned_files)} 个上传文件，成功导入 {len(loaded_files)} 个文件，"
                f"共 {expected_rows} 条记录，并写入本地数据库 {synced_rows} 条记录。"
            ),
            dataframe=dataframe,
            price_col=price_col or "",
            scanned_files=scanned_files,
            loaded_files=loaded_files,
            failed_files=failed_files,
            synced_rows=int(synced_rows or 0),
        )

    @staticmethod
    def _failed(report: FolderLoadReport, message: str) -> ImportJobResult:
        return ImportJobResult(
            success=False,
            message=message,
            dataframe=report.dataframe,
            price_col=report.price_col or "",
            scanned_files=report.scanned_files,
            loaded_files=report.loaded_files,
            failed_files=report.failed_files,
        )
