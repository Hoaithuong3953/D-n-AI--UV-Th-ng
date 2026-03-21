"""
prompts.py

Prompt templates and system instructions for the LearnPath Gemini-based assistant

Key features:
- SYSTEM_PROMPT: system instruction for chat behavior (Vietnamese, education-focused)
- ROADMAP_PROMPT_TEMPLATE: template for generating roadmap JSON from user profile
- PROFILE_EXTRACT_PROMPT: template for extracting UserProfile fields from chat history
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

PROFILE_EXTRACT_PROMPT = """
Từ đoạn hội thoại sau, trích xuất thông tin hồ sơ học tập từ tin nhắn của USER
Trả về ĐÚNG MỘT object JSON với các key sau:

REQUIRED (bắt buộc - phải có rõ ràng trong tin nhắn USER):
- goal: string (mục tiêu học tập, vd: "Học Python", "Lập trình web")
current_level: một trong "beginner", "intermediate", "advanced"
- time_commitment: string (thời gian mỗi ngày, vd: "30 phút", "1 giờ", "2 giờ")

OPTIONAL (nếu có thông tin từ USER):
- learning_style: string (phong cách học tập, vd: "Học qua video", "Đọc tài liệu", "Thực hành")
- background: string (nền tảng/kinh nghiệm trước đó, vd: "Đã học HTML/CSS", "Chưa biết lập trình")
- constraints: array of strings (các ràng buộc, vd: ["Chỉ tài liệu miễn phí", "Học vào cuối tuần"])

QUY TẮC QUAN TRỌNG:
1. CHỈ trích xuất thông tin từ tin nhắn của USER, BỎ QUA tin nhắn của Assistant
2. KHÔNG bịa hoặc suy đoán thông tin không có trong tin nhắn USER
3. Nếu USER không cung cấp đủ dữ kiện để trích xuất một REQUIRED key nào đó thì:
   - KHÔNG đưa key đó vào JSON (trả partial JSON)
4. Nếu trích xuất được ít nhất 1 REQUIRED key thì trả JSON chỉ chứa các key đã trích xuất được
5. Chỉ trả {} khi KHÔNG trích xuất được bất kỳ REQUIRED key nào
6. current_level:
   - "từ đầu", "mới bắt đầu", "beginner", "cơ bản", "người mới" → current_level = "beginner"
   - "intermediate", "đã có kinh nghiệm", "khá", "có nền" → current_level = "intermediate"
   - "advanced", "nâng cao", "chuyên sâu" → current_level = "advanced"
   - Nếu USER không nói rõ mức độ thuộc nhóm nào ở trên → OMIT key current_level
7. time_commitment chỉ là thời gian HỌC MỖI NGÀY.
   - Nếu USER nói deadline/timeline (vd: "trong 2 tháng") mà KHÔNG nói thời gian mỗi ngày → OMIT key time_commitment
8. Nếu có thông tin optional thì thêm vào JSON, không có thì bỏ qua (đừng thêm null/None)
9. Extract CHÍNH XÁC từ nguyên văn, không paraphrase nếu không cần thiết

FORMAT MẪU (chỉ để tham khảo):

Ví dụ 1 - Đầy đủ thông tin:
Hội thoại:
User: "Tôi muốn học Python cơ bản, mình mới bắt đầu, có khoảng 2 giờ/ngày"
User: "Tôi thích học qua video và đã biết HTML/CSS rồi"
Assistant: "Tốt, tôi sẽ tạo lộ trình..."

JSON:
{
    "goal": "Học Python cơ bản",
    "current_level": "beginner",
    "time_commitment": "2 giờ",
    "learning_style": "Học qua video",
    "background": "Đã biết HTML/CSS"
}

Ví dụ 2 - Chỉ có required:
Hội thoại:
User: "Học JavaScript cho người mới, 1 giờ mỗi ngày"
Assistant: "Bạn có muốn học qua video không?"
User: "Tùy cũng được"

JSON:
{
    "goal": "Học JavaScript",
    "current_level": "beginner",
    "time_commitment": "1 giờ"
}

Ví dụ 3 - Thiếu time_commitment (KHÔNG đủ required):
Hội thoại:
User: "Tôi muốn học React, mình intermediate rồi"
Assistant: "Bạn có bao nhiêu thời gian mỗi ngày?"
User: "Chưa biết, linh hoạt"

JSON:
{
    "goal": "Học React",
    "current_level": "intermediate"
}

Ví dụ 4 - Assistant nói sai, USER sửa (CHỈ lấy từ USER):
Hội thoại:
User: "Tôi muốn học Python"
Assistant: "Bạn muốn học advanced phải không?"
User: "Không, mình mới bắt đầu mà, beginner. Có 1 giờ/ngày"

JSON:
{
    "goal": "Học Python",
    "current_level": "beginner",
    "time_commitment": "1 giờ"
}
(KHÔNG extract "advanced" từ Assistant message!)

Ví dụ 5 - USER nói mơ hồ về level (KHÔNG bịa):
Hội thoại:
User: "Tôi muốn học Python, có 2 giờ/ngày"
Assistant: "Bạn đang ở level nào?"
User: "Cũng biết chút chút"

JSON:
{}
(USER không nói rõ "beginner"/"intermediate"/"advanced" -> KHÔNG đủ required!)

Hội thoại (chỉ xét các tin nhắn gần đây):
---
{history}
---
JSON:
"""