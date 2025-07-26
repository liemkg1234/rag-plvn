import requests
import pandas as pd
from enum import Enum

class PoliciesId(str, Enum):
    LuatBHXH_41 = "Luat_41_2024_QH15_20250726090104044522"
    LuatLaoDong_45 = "Luat_45_2019_QH14_20250726091858830992"
    LuatHinhSu_100 = "Luat_100_2015_QH13_20250726092055717518"
    LuatDieuKienLaoDong_145 = "Luat_145_2020_NĐ_CP_20250726092234596408"
    LuatAnToanGiaoThong_168 = "Luat_168_2024_NĐ_CP_20250726092337007149"


questions_data = [
    # 45
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Người lao động bị sa thải có được trả lương hay không?"},
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Người sử dụng lao động được sa thải người lao động nữ đang mang thai không?"},
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Quy định về điều chuyển nhân sự được quy định như thế nào?"},
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Làm việc 8h một ngày thì được nghỉ giữa giờ ít nhất bao nhiêu phút?"},
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Người sử dụng lao động đào tạo nghề nghiệp và phát triển kỹ năng nghề cho người lao động như thế nào?"},
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Nguyên tắc cho thuê lại lao động là gì?"},
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Thời hạn của thỏa ước lao động tập thể như thế nào?"},
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Hợp đồng lao động được giao kết theo hình thức nào?"},
    {"policy_id": PoliciesId.LuatLaoDong_45.value, "question": "Nội dung về đào tạo lao động có bắt buộc phải ghi vào hợp đồng lao động?"},

    # 145
    {"policy_id": PoliciesId.LuatDieuKienLaoDong_145.value, "question": "Người lao động được thuê làm giám đốc doanh nghiệp Nhà nước được hưởng các chế độ về tiền lương, thưởng như thế nào?"}
]
ground_truths = [
    "Điều 34, 48, 125",
    "Điều 137",
    "Điều 21, Điều 29",
    "Điều 105, 109, 18",
    "Điều 61, 62, 39",
    "Điều 53",
    "Điều 78, 83, 76",
    "Điều 13, 20",
    "Điều 21, 61",

    "Điều 5, 2, 101",

]

results = []

for question, gt in zip(questions_data, ground_truths):
    url = "http://localhost:8001/rag/retriever"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "question": question['question'],
        "collection_ids": [question['policy_id']]
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    json_data = response.json()

    predict = json_data.get("document_related", {}).get(question['policy_id'], "")

    results.append({
        "policy_id": question['policy_id'],
        "question": question['question'],
        "predict": predict,
        "ground_truth": gt
    })


df = pd.DataFrame(results)
df.to_csv("results.csv", index=True, encoding="utf-8-sig")
