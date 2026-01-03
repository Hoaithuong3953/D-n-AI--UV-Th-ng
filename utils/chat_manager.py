from typing import List, Dict
from datetime import datetime

class ChatManager:
    def __init__(self):
        self.max_message = 50

    def add_message(self, messages: List[Dict], role: str, content: str) -> List[Dict]:
        new_message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }

        messages.append(new_message)

        if len(messages) > self.max_message:
            messages = messages[-self.max_message:]

        return messages
    
    def get_recent_messages(self, messages: List[Dict], limit: int = 10) -> List[Dict]:
        return messages[-limit:] if messages else []
    
    def format_messages_for_ai(self, messages: List[Dict]) -> str:
        formatted = []
        for msg in messages[-5:]:
            role = 'User' if msg['role'] == 'user' else 'Assistant'
            formatted.append(f"{role}: {msg['content']}")

        return '\n'.join(formatted)
    
    def clear_history(self) -> None:
        return []