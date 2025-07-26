import asyncio

import streamlit as st
from api.rag import index_documents

st.set_page_config(page_title="Document Indexer")

st.title("📄 2. Index Documents (RAG)")


collection_name = st.text_input(
    "Collection Name",
    value="policy_id_xxx",
    help=(
        "Tên dùng để lưu collection, chỉ gồm chữ thường, số, gạch dưới. "
        "Ví dụ: `nghiquyet_42_2017_qh14` tương ứng với 'Nghị Quyết 42.2017.QH14'."
    )
)

description = st.text_input(
    "Description",
    value="Policy ID xxx - Policy Name - Group Policy Name",
    help="Mô tả chi tiết cho collection, giúp phân biệt nội dung lưu trữ."
)

uploaded_files = st.file_uploader(
    "Upload documents (.md)",
    type=["md"],
    accept_multiple_files=True
)

if st.button("🚀 Index Documents"):
    if not uploaded_files or not collection_name:
        st.warning("⚠️ Please provide a collection name and at least one file.")
    else:
        with st.spinner("Indexing in progress..."):
            res = asyncio.run(index_documents(collection_name, description, uploaded_files))

            if res.status_code == 200:
                json_data = res.json()
                returned_collection = json_data.get("collection_name")

                st.success("✅ Documents indexed successfully!")

                if returned_collection:
                    st.markdown(f"""
                    ### 📌 Indexed Collection
                    ```
                    {returned_collection}
                    ```
                    """)
                    st.info("💡 You can now query this collection.")
            else:
                st.error(f"❌ Failed: {res.status_code} - {res.text}")
