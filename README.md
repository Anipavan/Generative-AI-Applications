# 🚀 Resume ATS Score & Enhancement Tool

A smart, AI-powered Streamlit application that helps job applicants improve their resume by evaluating their match against a job description and intelligently updating the technical skills section for better Applicant Tracking System (ATS) results.

---

## 💡 Features

- **Upload Your Resume (PDF)**
- **Paste Any Job Description**
- **Automatic Extraction of Technical Skills (using LLM)**
- **AI-Based Matching & ATS Score Calculation**
- **Highlights Missing Technical Skills**
- **Intelligent Merge of New Skills into Your Resume**
- **Download Enhanced Resume (TXT & PDF)**

---

## 🎯 How It Works

1. **Upload Resume:** Drop your resume PDF into the app.
2. **Paste Job Description:** Provide details from the job description.
3. **Run ATS Check:** Click to see your ATS match score and missing skills.
4. **Get Enhanced Resume:** Download a version with new technical skills seamlessly integrated under your existing skills section.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – UI & Webapp Framework
- [LangChain](https://www.langchain.com/) – LLM Document Processing
- [OpenAI GPT-4o](https://platform.openai.com/docs/guides/gpt) – Skill Extraction & Matching
- [FAISS](https://github.com/facebookresearch/faiss) – Semantic Search
- [PyPDF2](https://pypdf2.readthedocs.io/) – PDF Text Extraction
- [ReportLab](https://www.reportlab.com/) – PDF File Generation
- [Scikit-learn](https://scikit-learn.org/) – Cosine Similarity
- [python-dotenv](https://pypi.org/project/python-dotenv/) – Secure API Key Management

---

## ⚡ Installation

Clone the repository
git clone https://github.com/Anipavan/Generative-AI-Applications.git
cd Generative-AI-Applications

Install required packages
pip install -r requirements.txt


---

## 🔑 Setup Environment Variables

1. **Create a `.env` file in your project directory**
2. **Add your OpenAI API key:**

    ```
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
    ```


## 🚀 Launch the App

streamlit run app.py


---

## 📝 Usage

- Visit `http://localhost:8501` (default Streamlit port)
- Upload your resume (PDF format)
- Paste the job description
- Click "Run ATS"
- Review your ATS score, preview the enhanced resume
- Download your improved resume as TXT or PDF

---


## 💬 FAQ

**Q: Does this app store my resume or personal data?**  
A: No, all processing is done in-memory; no files are stored.

**Q: Is my API key safe?**  
A: Yes, use `.env` locally and Streamlit Cloud secrets in production. Keys are never published.

**Q: Can I use this for any resume & any job description?**  
A: Yes! The app is designed to handle all standard resume PDFs and job descriptions.

---

## 👏 Contributions

Pull requests, bug reports, and feature suggestions are welcome!

---


## 🤖 Author
 Anirudh

