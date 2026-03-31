
import streamlit as st

def init_session():
    st.session_state.setdefault("liked_idx", set())
    st.session_state.setdefault("blocked_idx", set())
    st.session_state.setdefault("bookmarked_idx", set())

def like_item(gid: int):
    st.session_state.liked_idx.add(int(gid))
    st.session_state.blocked_idx.discard(int(gid))

def skip_item(gid: int):
    st.session_state.blocked_idx.add(int(gid))
    st.session_state.liked_idx.discard(int(gid))

def toggle_bookmark(gid: int):
    gid = int(gid)
    if gid in st.session_state.bookmarked_idx:
        st.session_state.bookmarked_idx.remove(gid)
    else:
        st.session_state.bookmarked_idx.add(gid)

def clear_feedback():
    st.session_state.liked_idx.clear()
    st.session_state.blocked_idx.clear()
