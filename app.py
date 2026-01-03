import streamlit as st
from config.config import APP_CONFIG
import datetime
from utils.chat_manager import ChatManager
from utils.helpers import validate_user_input, clean_text

st.set_page_config(
    page_title=APP_CONFIG['name'],
    layout='wide'
)

st.markdown("""
<style>
.chat-interface {
    max-height: 600px;
    overflow-y: auto;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 10px;
    margin: 10px 0;
}
.user-message {
    background-color: #e3f2fd;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
}
.bot-message {
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title(f'{APP_CONFIG["name"]}')
st.markdown('Chatbot AI tạo lộ trình học tập cá nhân hoá')

# Sidebar
with st.sidebar:
    st.header('Thông tin')
    st.info(f'Version: {APP_CONFIG["version"]}')

    if st.button('Xoá lịch sử chat'):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success('Đã xoá lịch sử')

# Chat interface
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f'<div class="user-message"> **Bạn:** {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"> **LearnPath:** {message["content"]}</div>', unsafe_allow_html=True)

# Chat input form
with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_input(
        "Nhập mục tiêu học của bạn:",
        placeholder="Ví dụ: Tôi muốn học Python trong 3 tháng..."
    )

    chat_manager = ChatManager()

    submitted = st.form_submit_button('Gửi', type='primary')
    if submitted and user_input.strip():
        cleaned_input = clean_text(user_input)
        validation = validate_user_input(cleaned_input)

        if validation['is_valid']:
            st.session_state.messages = chat_manager.add_message(st.session_state.messages, 'user', cleaned_input)

            bot_response = f'Tôi hiểu bạn muốn "{cleaned_input}". Roadmap sẽ được tạo...'
            st.session_state.messages = chat_manager.add_message(st.session_state.messages, 'bot', bot_response)

            st.success(f"{validation['message']} (Confidence: {validation['confidence']:.1f})")
            st.rerun()
        else:
            st.error(f"{validation['message']}")