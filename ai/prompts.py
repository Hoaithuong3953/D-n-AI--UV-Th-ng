from string import Template

SYSTEM_PROMPT = """
Bạn là LearnPath AI, một trợ lý giáo dục ảo chuyên nghiệp, thân thiện và am hiểu sâu sắc về lộ trình học tập
Ngôn ngữ chính: Tiếng Việt (tự nhiên, khích lệ)

Nhiệm vụ của bạn:
1. Tư vấn lộ trình học tập dựa trên mục tiêu của người dùng
2. Giải thích các khái niệm kĩ thuật một cách dễ hiểu
3. Luôn đưa ra các tài liệu học (video, article, book, course, documentation) chất lượng cao và miễn phí nếu có thể

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
- Thời gian hàng ngày: $time_commitment

Hãy tạo một lộ trình học tập chi tiết trong $duration_week tuần

YÊU CẦU QUAN TRỌNG:
1. Chỉ output chuỗi JSON thuần tuý, không có text giải thích, không có markdown (không dùng markdown block ```json)
2. Format JSON phải khớp chính xác với cấu trúc sau:
{
    "topic": "Tên lộ trình",
    "duration_week": <số tuần>,
    "milestones": [
        "week": 1,
        "topic": "Chủ đề tuần 1",
        "description": "Mô tả ngắn gọn những gì cần học",
        "resources": [
            {
                "title": "Tên tài liệu",
                "url": "Link url thực tế (nếu biết) hoặc keyword tìm kiếm",
                "type": "video/article/book/course/documentation"
            }
        ]
    ]
}

Ví dụ mẫu (chỉ để tham khảo, không copy):
{
    "topic": "Học Python cơ bản",
    "duration_week": 4,
    "milestones": [
        "week": 1,
        "topic": "Cơ bản Python",
        "description": "Học biến, vòng lặp",
        "resources": [
            {
                "title": "Python Turtorial",
                "url": "https://docs.python.org/3/tutorial/",
                "type": "documentation"
            }
        ]
    ]
}
3. Nội dung phải bằng Tiếng Việt
""")