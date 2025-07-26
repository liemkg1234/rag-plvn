import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="<KEY>",
)

input = """
## Điều 45. Trợ cấp ốm đau

1. Mức hưởng trợ cấp ốm đau được tính theo tháng và tính trên căn cứ sau đây:

a) Tiền lương làm căn cứ đóng bảo hiểm xã hội của tháng gần nhất trước tháng nghỉ việc hưởng chế độ ốm đau;

b) Tiền lương làm căn cứ đóng bảo hiểm xã hội của tháng đầu tiên tham gia bảo hiểm xã hội hoặc tháng tham gia trở lại nếu phải nghỉ việc hưởng chế độ ốm đau ngay trong tháng đầu tiên tham gia hoặc tháng tham gia trở lại.

2. Mức hưởng trợ cấp ốm đau của người lao động quy định tại khoản 1 Điều 43 và Điều 44 của Luật này bằng 75% tiền lương làm căn cứ đóng bảo hiểm xã hội quy định tại khoản 1 Điều này.

3. Mức hưởng trợ cấp ốm đau của người lao động quy định tại khoản 2 Điều 43 của Luật này được tính như sau:

a) Bằng 65% tiền lương làm căn cứ đóng bảo hiểm xã hội quy định tại khoản 1 Điều này nếu đã đóng bảo hiểm xã hội bắt buộc từ đủ 30 năm trở lên;

b) Bằng 55% tiền lương làm căn cứ đóng bảo hiểm xã hội quy định tại khoản 1 Điều này nếu đã đóng bảo hiểm xã hội bắt buộc từ đủ 15 năm đến dưới 30 năm;

c) Bằng 50% tiền lương làm căn cứ đóng bảo hiểm xã hội quy định tại khoản 1 Điều này nếu đã đóng bảo hiểm xã hội bắt buộc dưới 15 năm.

4. Mức hưởng trợ cấp ốm đau của người lao động quy định tại khoản 3 Điều 43 của Luật này bằng 100% tiền lương làm căn cứ đóng bảo hiểm xã hội quy định tại khoản 1 Điều này.

5. Mức hưởng trợ cấp ốm đau một ngày được tính bằng mức hưởng trợ cấp ốm đau theo tháng chia cho 24 ngày. Mức hưởng trợ cấp ốm đau nửa ngày được tính bằng một nửa mức hưởng trợ cấp ốm đau một ngày.
Khi tính mức hưởng trợ cấp ốm đau đối với người lao động nghỉ việc hưởng chế độ ốm đau không trọn ngày thì trường hợp nghỉ việc dưới nửa ngày được tính là nửa ngày; từ nửa ngày đến dưới một ngày được tính là một ngày.

6.

"""

response = client.embeddings.create(
    model="bge-m3",
    input=input,
    encoding_format="float",
)

print(response.data[0].embedding)
print(len(response.data[0].embedding))
