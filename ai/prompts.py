"""
Prompt templates and system instructions used by the LearnPath Gemini-based assistant.
"""

from string import Template

SYSTEM_PROMPT = """
Bạn là LearnPath AI, một trợ lý giáo dục ảo chuyên nghiệp, thân thiện và am hiểu sâu sắc về lộ trình học tập
Ngôn ngữ chính: Tiếng Việt (tự nhiên, khích lệ)

Nhiệm vụ của bạn:
1. Tư vấn lộ trình học tập dựa trên mục tiêu của người dùng
2. Giải thích các khái niệm kĩ thuật một cách dễ hiểu
3. Luôn đưa ra các tài liệu học (video, article, book, course, documentation,...) chất lượng cao và miễn phí nếu có thể

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
- Phong cách học: $learning_style
- Nền tảng: $background
- Ràng buộc: $constraints

Hãy tạo một lộ trình học tập chi tiết trong $duration_week tuần

YÊU CẦU QUAN TRỌNG:
1. Chỉ output chuỗi JSON thuần tuý, không có text giải thích, không có markdown (không dùng markdown block ```json)
2. Format JSON phải khớp CHÍNH XÁC với cấu trúc sau:
{
    "topic": "Tên lộ trình",
    "title": "Tiêu đề hiển thị (nếu có)",
    "description": "Mô tả ngắn gọn những gì cần học (nếu có)",
    "duration_week": <số tuần>,
    "prerequisites": ["Yêu cầu tiên quyết (nếu có, danh sách các mục tiêu cần đạt trước khi bắt đầu)"],
    "milestones": [
        {
            "week": <số tuần>,
            "topic": "Chủ đề tuần <số tuần>",
            "description": "Mô tả chi tiết những gì cần học trong tuần <số tuần>",
            "estimated_time": "Thời gian ước tính cho tuần <số tuần> (nếu có)",
            "learning_objectives": ["Mục tiêu học tập (nếu có, danh sách các mục tiêu cần đạt trong tuần <số tuần>)"],
            "resources": [
                {
                    "title": "Tên tài liệu",
                    "url": "https://example.com",
                    "type": "video | article | book | course | practice | project | documentation",
                    "description": "Mô tả tài liệu (nếu có)",
                    "difficulty": "beginner | intermediate | advanced"
                }
            ]
        }
    ]

Ví dụ mẫu (chỉ để tham khảo, không copy):
{
    "topic": "Học Python cơ bản",
    "title": "Lộ trình học Python cơ bản",
    "description": "Lộ trình học Python cơ bản cho người mới bắt đầu",
    "duration_week": 4,
    "prerequisites": ["Kiến thức cơ bản về máy tính", "Có thể sử dụng terminal"],
    "milestones": [
        {
            "week": 1,
            "topic": "Cơ bản Python",
            "description": "Học biến, vòng lặp",
            "estimated_time": "5 giờ",
            "learning_objectives": ["Hiểu về biến và kiểu dữ liệu", "Sử dụng if/else, for, while", "Làm quen với Python syntax"],
            "resources": [
                {
                    "title": "Python Turtorial",
                    "url": "https://docs.python.org/3/tutorial/",
                    "type": "documentation",
                    "description": "Tài liệu chính thức của Python",
                    "difficulty": "beginner"
                }
            ]
        },
        ...
    ]
}
3. Ràng buộc validation:
- Số tuần trong milestones PHẢI khớp với duration_week
- week trong milestones PHẢI là số nguyên dương và tăng dần từ 1 đến duration_week
- Mỗi milestone PHẢI có ít nhất 1 resource
4. Nội dung phải bằng Tiếng Việt
"""
)