import re
from typing import Dict, Any
import datetime

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text.strip())

    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]', ' ', text)

    return text.strip()

def validate_user_input(text: str) -> Dict[str, Any]:
    result = {
        'is_valid': False,
        'message': '',
        'confidence': 0.0
    }

    if not text or len(text.strip()) < 5:
        result['message'] = 'Vui lòng nhập ít nhất 5 kí tự'
        return result
    
    learning_keywords = [
        'học', 'muốn học', 'cần học', 'bắt đầu học',
        'python', 'javascript', 'web', 'ai', 'machine learning', 
        'data', 'design', 'sql', 'git', 'docker'
    ]

    text_lower = text.lower()
    matched_keywords = [kw for kw in learning_keywords if kw in text_lower]

    if matched_keywords:
        result['is_valid'] = True
        result['confidence'] = min(len(matched_keywords) * 0.2 + 0.3, 1.0)
        result['message'] = 'Input hợp lệ'
    else:
        result['message'] = 'Vui lòng mô tả mục tiêu học tập cụ thể hơn'

    return result

def format_timestamp(timestamp: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%H:%M')
    except:
        return timestamp
    
def truncate_text(text: str, max_length: int=100) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."