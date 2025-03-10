import requests
import json
import os
from typing import List, Dict, Any

# Set up Groq API key
GROQ_API_KEY = 'YOUR_TOKEN_HERE' # Thay thế bằng API key thực của bạn

def generate_absa_data(num_samples: int, tag_labels: List[str]) -> List[Dict[str, Any]]:
    prompt = f"""
    Please create a dataset for Aspect-Based Sentiment Analysis (ABSA) in Vietnamese, following the format below:
    ```json
    {{"text": "<Vietnamese text>", "labels": [[<start_position>, <end_position>, "<TAG>"]]}}
    ```
    **Requirements:**
    - **text**: A Vietnamese sentence or paragraph that reflects a user's opinion about a smartphone product or service.
    - **labels**: A list of labels, each label includes:
      - **start_position**: The character index where the aspect term starts.
      - **end_position**: The character index where the aspect term ends.
      - **TAG**: A label in the format "ASPECT#SENTIMENT", for example: "GENERAL#POSITIVE".
    Use the following tag labels: {tag_labels}.
    **Examples:**
    ```jsonl
    {{"text": "Bắt sóng quá yếu k nổi 1 vạch4g Viettel chỉ 1 vạch đứng dưới cột phát sóng vẫn bắt yếu", "labels": [[0, 86, "FEATURES#NEGATIVE"]]}}
    {{"text": "Máy đẹp cấu hình ngon.. sạc rất nhanh... Ưng nhất cái camera lên ảnh đẹp thôi r... Mở khoá vân tay cực nhậy.. đúng kiểu ngon-bổ-rẻ", "labels": [[0, 7, "DESIGN#POSITIVE"], [8, 21, "PERFORMANCE#POSITIVE"], [24, 37, "BATTERY#POSITIVE"], [41, 79, "CAMERA#POSITIVE"], [83, 107, "FEATURES#POSITIVE"], [120, 127, "GENERAL#POSITIVE"], [128, 130, "PRICE#POSITIVE"]]}}
    {{"text": "vừa mới mua hôm qua đã test khá kỹ pin và tác vụ xài ổn, tuy nhiên có cái thấy máy mà hộp đã khui ra sẵn với máy đầu tiên nhân viên mang ra bị trầy camera sau đổi lại máy khác thì cũng là hộp đã khui rồi cảm giác như đang xài máy cũ :))còn ngoài ra anh em nào cần máy chơi game thì máy này xài rất oke nhé pin ổn", "labels": [[35, 38, "BATTERY#POSITIVE"], [42, 55, "PERFORMANCE#POSITIVE"], [79, 232, "SER&ACC#NEGATIVE"], [264, 305, "PERFORMANCE#POSITIVE"], [306, 312, "BATTERY#POSITIVE"]]}}
    {{"text": "Từ lúc mới mua, mỗi khi sạc nó hiện cái vòng tròn sạc nhanh , sài đc tầm 7-8 tháng. Bây giờ nó mất tiêu, sạc như sạc thông thường không còn đc sạc nhanh nữa Tại sao? Hay do nó vẫn sạc nhanh, chỉ là mất cái vòng tròn thôi.", "labels": [[92, 156, "BATTERY#NEGATIVE"]]}}
    {{"text": "Mua về là vừa xạc vừa chơi luôn Choi game hơi tụt pin lượt wer ok xem phim oke", "labels": [[32, 53, "BATTERY#NEGATIVE"], [54, 79, "PERFORMANCE#POSITIVE"]]}}
    {{"text": "Ban đầu mới mua tất cả đều ok. Sau này chụp ảnh nhiều mới phát hiện ra ,nếu chụp đồ ăn gần ,nó chỉ lấy nét ở chính giữa. Xung quanh bị mờ. Chụp xa thì không sao. Khác xa với Huawei. Chụp ảnh thức ăn vẫn nét tất cả", "labels": [[72, 160, "CAMERA#NEGATIVE"]]}}
    {{"text": "Máy quá ngon phải nói là khoẻ nhất phân khúc 4 củ quá toàn diện. Pin trâu. Chip khoẻ. Xạc nhanh .chiến game quá tốt nói chung là ngon 5 *", "labels": [[0, 63, "GENERAL#POSITIVE"], [65, 73, "BATTERY#POSITIVE"], [75, 84, "PERFORMANCE#POSITIVE"], [86, 95, "BATTERY#POSITIVE"], [97, 116, "PERFORMANCE#POSITIVE"], [116, 133, "GENERAL#POSITIVE"]]}}
    {{"text": "Mình thích kiểu iPhone 5s nhưng màn hình lớn hơn chút thì Ok, cấu hình thì được nhưng thiết kế không đẹp bằng 5s, đó tận dụng linh kiện từ iPhone 8, dung lượng pin hơi thấp nhưng điều đó không còn quá quan trọng khi được trang bị sạc nhanh", "labels": [[32, 60, "SCREEN#NEGATIVE"], [62, 79, "PERFORMANCE#POSITIVE"], [86, 112, "DESIGN#NEGATIVE"], [149, 172, "BATTERY#NEGATIVE"], [216, 239, "BATTERY#POSITIVE"]]}}
    {{"text": "Máy xài rất tốt, chụp ảnh đẹp, ko bị lỗi lầm gì, thiết kế bằng kính quá đẹp, vân tay nhạy", "labels": [[0, 15, "GENERAL#POSITIVE"], [17, 29, "CAMERA#POSITIVE"], [31, 47, "GENERAL#POSITIVE"], [49, 75, "DESIGN#POSITIVE"], [77, 89, "FEATURES#POSITIVE"]]}}
    {{"text": "Chụp cận cảnh không đẹp như quảng cáo. Mở chế độ cận cảnh lên là màn hình ruồi không, chụp gần mờ chứ ko rõ nét. Hay do máy mình lỗi. Sạc nhanh thì bị nóng máy. Khi mua thì thấy máy đc dán màn hình sẵn rồi, hộp cũng bị rách tem niêm phong rồi nhưng nhân viên vẫn đảm bảo là máy mới.", "labels": [[0, 111, "CAMERA#NEGATIVE"], [134, 159, "BATTERY#NEGATIVE"], [161, 281, "SER&ACC#NEGATIVE"]]}}
    {{"text": "Máy quá kém. Phí 8 triệu. Đk mỗi cái mã ngoài. Máy hay lag, đơ. Bắt sóng kém. Không đáng mua chút nào.", "labels": [[0, 11, "GENERAL#NEGATIVE"], [13, 24, "PRICE#NEGATIVE"], [26, 45, "DESIGN#POSITIVE"], [47, 62, "PERFORMANCE#POSITIVE"], [64, 76, "FEATURES#POSITIVE"], [78, 101, "GENERAL#NEGATIVE"]]}}
    {{"text": "Đt chơi game mượt,pin tụt khá nhanh nhưng có hỗ trợ sạc nhanh nên không sao, với tầm giá này thì Ok ! Với lại cách nhân viên phục vụ rất tận tình ❤️Không như nyc 😌", "labels": [[0, 17, "PERFORMANCE#POSITIVE"], [18, 75, "BATTERY#NEUTRAL"], [81, 99, "PRICE#POSITIVE"], [110, 144, "SER&ACC#POSITIVE"]]}}
    {{"text": "Hãng Xiaomi này.thua HAWAI nha.may Xiaomi redmi note 8pro nay mua về chơi pubg thời Gian đầu chơi khoản hết 1doi 1 là xong nha thời gian còn lại là chơi khoản 4g là đập quả trứng gà lên màng hình là chín luôn đó tui nói không sai đâu", "labels": [[148, 211, "PERFORMANCE#NEGATIVE"]]}}
    {{"text": "Máy làm j cũng ngon chiến game tốt chụp hình nét lưu trử tốt ai thích mua hãng khác nhưng riêng tôi lại tin tưởng hàng Việt Nam chất lượng cao.", "labels": [[0, 19, "GENERAL#POSITIVE"], [19, 34, "PERFORMANCE#POSITIVE"], [34, 48, "CAMERA#POSITIVE"], [48, 60, "STORAGE#POSITIVE"], [104, 142, "SER&ACC#POSITIVE"]]}}
    {{"text": "Máy cũng xài tạm ổn. Màn hình nó ko được nhay nhiều lúc phải bấm vài lần mới được.", "labels": [[0, 24, "GENERAL#NEUTRAL"], [26, 97, "FEATURES#NEGATIVE"]]}}
    ```
    **Your Task:**
    - Generate {num_samples} sentences or paragraphs following the above format with creative varieties.
    - Include opinions on aspects like design, screen, performance, battery, camera, features, price, and service
    - Each sentence must contain at least one aspect from the provided tag labels but should contain multiple aspects (3 or more), as many as possible.
    - Ensure the start and end positions accurately correspond to the phrases in the "text".
    - Use colloquial Vietnamese expressions. The content should be natural, can be informal or even swear, but accurately reflect the sentiment associated with each aspect.
    - Can mention real-life usage experience or comparison with other products.
    - Include occasional typos or informal spellings
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-70b-versatile",  # Thay thế bằng model Groq phù hợp
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates ABSA data in Vietnamese."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0,
        "max_tokens": 5000
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response_json = response.json()

    generated_data = response_json['choices'][0]['message']['content']
    
    # Parse the generated data
    data_list = []
    for line in generated_data.split('\n'):
        if line.strip().startswith('{') and line.strip().endswith('}'):
            try:
                data = json.loads(line.strip())
                data_list.append(data)
            except json.JSONDecodeError:
                print(f"Error parsing line: {line}")

    return data_list

# Example usage
tag_labels = ["BATTERY#NEGATIVE","BATTERY#NEUTRAL","BATTERY#POSITIVE","CAMERA#NEGATIVE","CAMERA#NEUTRAL","CAMERA#POSITIVE","DESIGN#NEGATIVE","DESIGN#NEUTRAL","DESIGN#POSITIVE","FEATURES#NEGATIVE","FEATURES#NEUTRAL","FEATURES#POSITIVE","GENERAL#NEGATIVE","GENERAL#NEUTRAL","GENERAL#POSITIVE","PERFORMANCE#NEGATIVE","PERFORMANCE#NEUTRAL","PERFORMANCE#POSITIVE","PRICE#NEGATIVE","PRICE#NEUTRAL","PRICE#POSITIVE","SCREEN#NEGATIVE","SCREEN#NEUTRAL","SCREEN#POSITIVE","SER&ACC#NEGATIVE","SER&ACC#NEUTRAL","SER&ACC#POSITIVE","STORAGE#NEGATIVE","STORAGE#NEUTRAL","STORAGE#POSITIVE"]
num_samples = 20

generated_data = generate_absa_data(num_samples, tag_labels)

def append_generated_data(generated_data, output_file_path):
    with open(output_file_path, 'a', encoding='utf-8') as f:
        for item in generated_data:
            # Ensure the item has the correct format
            formatted_item = {
                "text": item["text"],
                "labels": [[label[0], label[1], label[2]] for label in item["labels"]]
            }
            json_line = json.dumps(formatted_item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"Data has been appended to {output_file_path}")

# Specify the output file path
output_file_path = 'generated_data_2.0.jsonl'

# Append the generated data
append_generated_data(generated_data, output_file_path)