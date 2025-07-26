# isort: skip_file

import asyncio

import streamlit as st
from api.rag import get_qdrant_collections, retriever

import streamlit_nested_layout

st.set_page_config(page_title="Retrieval", layout="wide")

st.title("📄 3. Retrival (RAG)")

# History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Sidebar
st.sidebar.title("Collections Available")
mode = st.sidebar.toggle('Chat mode', value=False)

@st.cache_data(ttl=5, show_spinner="Loading collections...")
def fetch_collections_sync():
    return asyncio.run(load_collections())

async def load_collections():
    try:
        data = await get_qdrant_collections()
        return data
    except Exception as e:
        st.sidebar.error(f"Error collections: {e}")
        return []

collections = fetch_collections_sync()

if collections:
    collection_names = [c["collection_name"] for c in collections]
    descriptions = {
        c["collection_name"]: c.get("description", "No description")
        for c in collections
    }
    name_to_id = {c["collection_name"]: c["id"] for c in collections}  # Mapping thêm

    selected_collections = st.sidebar.multiselect(
        "Choose collections to retrieve:",
        collection_names,
        help="Select one or more collections to retrieve documents from.",
    )

    if selected_collections:
        st.sidebar.markdown("### 📌 Description of collections:")
        for name in selected_collections:
            st.sidebar.caption(f"• **{name}**: {descriptions[name]}")

        selected_ids = [name_to_id[name] for name in selected_collections]
    else:
        st.sidebar.info("Don't have any collection selected.")

    # Chat Input
    user_input = st.chat_input("...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving collection..."):
                try:
                    response = asyncio.run(
                        retriever(
                            selected_ids,
                            st.session_state.chat_history,
                            mode
                        )
                    )
                    if not mode:
                        assistant_reply = "This is retrieval mode, don't have answer."
                    else:
                        assistant_reply = response['answer']

                    # Expander
                    document_related = response['document_related']
                    with st.expander(
                        "📚 Documents Related: "
                    ):
                        for collection_name, content in document_related.items():
                            with st.expander(f"📁 {collection_name}"):
                                st.code(content.strip(), language="markdown")

                except Exception as e:
                    assistant_reply = f"❌ Error: {e}"

                st.markdown(assistant_reply)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_reply}
                )

else:
    st.sidebar.warning("Don't have any collection available.")
