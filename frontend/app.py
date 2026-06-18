import requests
import streamlit as st
import os

st.title("🤖 Enterprise RAG Platform")

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True
)

question = st.text_input(
    "Ask a Question"
)

saved_files = []

if uploaded_files:

    for uploaded_file in uploaded_files:

        save_path = os.path.join(
            "data",
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        saved_files.append(
            save_path
        )

    st.success(
        f"Uploaded {len(saved_files)} file(s)"
    )

if st.button("Submit"):

    if uploaded_files and question:

        with st.spinner(
            "Thinking..."
        ):

            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={
                    "question": question
                }
            )

            

            if response.status_code == 200:

                result = response.json()

                answer = result["answer"]

                sources = result["sources"]

                st.subheader(
                    "Answer"
                )

                st.write(
                    answer
                )

                st.subheader(
                    "📚 Sources"
                )

                for source in sources:

                    st.write(
                        f"📄 {source['source']} | Chunk {source['chunk_id']}"
                    )

            else:

                st.error(
                    "FastAPI returned an error"
                )

    else:

        st.warning(
            "Please upload a document and enter a question."
        )