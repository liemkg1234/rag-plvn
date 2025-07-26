import streamlit as st

st.set_page_config(page_title="Parse Document", page_icon="📄")

st.title("📄 2. Parse Document (RAG)")
st.info(
    "Để parse tài liệu, vui lòng truy cập trang giao diện chuyên dụng tại "
    "[http://localhost:9999/ui/](http://localhost:9999/ui/)",
    icon="🔗"
)

st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNzQyZmFkZDNhYjBmYzk5NzViZDEwMjY1MmM5ODRjMmIxMDA4MmQ2NyZjdD1n/3oriO0OEd9QIDdllqo/giphy.gif",
         caption="Chuyển hướng thôi nào!", use_column_width=True)