from pathlib import Path


def resolve_report_path(row: dict[str, str], project_root: Path) -> Path:
    """优先使用仓库内的标准位置，兼容已有清单的本机路径。"""
    portable = project_root / "data" / "raw" / row["file_name"]
    if portable.is_file():
        return portable
    configured = row.get("local_path", "")
    if not configured:
        return portable
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def resolve_chunks_path(metadata: dict, metadata_path: Path) -> Path:
    path = Path(metadata["chunks_path"])
    if not path.is_absolute():
        return metadata_path.parent / path
    if path.is_file():
        return path
    # 兼容搬迁前写入绝对路径的旧版索引。
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == ("data", "processed"):
            return metadata_path.parent.parent / Path(*parts[index + 1:])
    return path
