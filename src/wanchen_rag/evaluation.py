import hashlib


API_FAILURE_TYPES = {
    "APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError",
    "AuthenticationError", "PermissionDeniedError",
}


def generation_succeeded(record: dict | None) -> bool:
    if not record or not str(record.get("answer", "")).strip():
        return False
    if record.get("generation_error") or record.get("generation_status") == "failed":
        return False
    # 旧文件的 error_type 同时存放接口失败和人工评价，不能把部分正确当作请求失败。
    if record.get("error_type") in API_FAILURE_TYPES:
        return False
    return record.get("manual_evaluation") != "待重新生成"


def evaluation_matches(answer: dict, evaluation: dict | None) -> bool:
    if not evaluation or not generation_succeeded(answer):
        return False
    if "generation_status" not in answer:
        return True  # 兼容仓库原有的人工评测记录。
    digest = hashlib.sha256(str(answer["answer"]).encode("utf-8")).hexdigest()
    return evaluation.get("answer_sha256") == digest
