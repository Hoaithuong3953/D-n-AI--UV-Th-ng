"""
header.py

Fixed header for LearnPath Chatbot main page

Key features:
- Render title and subtitle
- Provide "Bắt đầu phiên mới" button to reset session and rerun page
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from services import AppService

def render_header(app: AppService) -> None:
    """
    Render header with title, subtitle and reset-session button

    Args:
        app: AppService instance used to reset session when button is clicked
    """
    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.markdown("## LearnPath Chatbot")
        st.caption("Trợ lý AI tạo lộ trình học tập cá nhân hóa")
    with col_btn:
        if st.button("Bắt đầu phiên mới", use_container_width=True):
            app.reset_session()
            st.rerun()
    st.divider()