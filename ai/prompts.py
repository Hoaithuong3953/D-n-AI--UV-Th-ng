from string import Template

SYSTEM_PROMPT = """
Bạn là LearnPath AI, một trợ lý giáo dục ảo chuyên nghiệp, thân thiện và am hiểu sâu sắc về lộ trình học tập
Ngôn ngữ chính: Tiếng Việt (tự nhiên, khích lệ)

Nhiệm vụ của bạn:
1. Tư vấn lộ trình học tập dựa trên mục tiêu của người dùng
2. Giải thích các khái niệm kĩ thuật một cách dễ hiểu
3. Luôn đưa ra các tài liệu học (docs, video, course) chất lượng cao và miễn phí nếu có thể

Quy tắc ứng xử:
- Không trả lời các câu hỏi không liên quan đến giáo dục/học tập
- Nếu không chắc chắn, hãy nói rõ là bạn cần thêm thông tin
- Luôn giữ thái độ tích cực, động viên người học
"""

ROADMAP_PROMPT_TEMPLATE = Template(
"""
Dựa trên thông tin sau của người dùng:
- Mục tiêu: $goal
- Trình độ hiện tại: $level
- Thời gian cam kết: $time_commitment

Hãy tạo một lộ trình học tập chi tiết trong $duration_weeks
YÊU CẦU QUAN TRỌNG:
1. Output phải là chuỗi JSON hợp lệ (không có markdown block ```json)
2. Format JSON phải khớp chính xác với cấu trúc sau:
{
    "topic": "Tên lộ trình",
    "duration_weeks": <số tuần>,
    "milistones": [
        "week": 1,
        "topic": "Chủ đề tuần 1",
        "describtion": "Mô tả ngắn gọn những gì cần học",
        "resources": [
            {
                "title": "Tên tài liệu",
                "url": "Link url thực tế (nếu biết) hoặc keyword tìm kiếm",
                "type": "video/article/course"
            }
        ]
    ]
}
3. Nội dung phải bằng Tiếng Việt
""")