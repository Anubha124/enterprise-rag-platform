import os

import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)


class GeminiClient:

    def __init__(self):

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
            )

    def generate_answer(
        self,
        question,
        context
    ):

        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = self.model.generate_content(
            prompt
        )

        return response.text


if __name__ == "__main__":

    client = GeminiClient()

    answer = client.generate_answer(
        "What is Anubha's CGPA?",
        """
        Anubha Khatri completed B.Tech in Computer Science Engineering at KIIT University.

        CGPA: 7.30 / 10
        """
    )

    print(answer)