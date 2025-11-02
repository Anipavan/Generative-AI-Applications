import os
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()


def documentLoader(document):
    _reader = PdfReader(document)
    text = ""
    for page in _reader.pages:
        text += page.extract_text()
    return text


def textSplitter(text):
    _text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = _text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs


def getEmbeddings():
    return OpenAIEmbeddings()


def getLLMModel():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def getDB(content):
    return FAISS.from_documents(content, getEmbeddings())


def extractSkillList(text):
    parts = re.split(r"[,;\n\-•:]+", text.lower())
    skills = [
        re.sub(r"[^a-z0-9+#.\s]", "", skill).strip()
        for skill in parts
        if len(skill.strip()) > 1
    ]
    return sorted(set(skills))


def calculateATS(resume_result, JD_result):
    matched_pairs = []
    jd_matched = set()
    resume_skills = extractSkillList(resume_result["result"])
    jd_skills = extractSkillList(JD_result["result"])
    st.write("Resume Skills:", resume_skills)
    st.write("Job Description Skills:", jd_skills)

    resume_vectors = getEmbeddings().embed_documents(resume_skills)
    jd_vectors = getEmbeddings().embed_documents(jd_skills)
    similarity_matrix = cosine_similarity(resume_vectors, jd_vectors)

    threshold = 0.85
    for i, r_skill in enumerate(resume_skills):
        for j, j_skill in enumerate(jd_skills):
            sim = similarity_matrix[i][j]
            if sim >= threshold:
                matched_pairs.append((r_skill, j_skill, round(sim, 2)))
                jd_matched.add(j_skill)

    missing_skills = [skill for skill in jd_skills if skill not in jd_matched]
    match_score = round((len(jd_matched) / len(jd_skills)) * 100, 2) if jd_skills else 0
    st.write("Missing Skills:", missing_skills)
    return match_score, missing_skills


def findTechnicalSkillsSection(resume_text):
    patterns = [
        r'(technical\s+skills|skills|technical\s+expertise|core\s+competencies|technologies)(\s*[:|\-|•]?\s*\n)',
        r'(technical\s+skills|skills|technical\s+expertise|core\s+competencies|technologies)(\s*[:|\-|•]?\s*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.end()
    
    return None


def addMissingSkillsToResume(resume_text, missing_skills):
    if not missing_skills:
        return resume_text
    

    skills_section_end = findTechnicalSkillsSection(resume_text)
    
    if skills_section_end is not None:
        next_section = re.search(r'\n\n[A-Z]', resume_text[skills_section_end:])
        
        if next_section:
            insert_position = skills_section_end + next_section.start()
        else:
            lines_after = resume_text[skills_section_end:].split('\n')
            skill_lines = 0
            for i, line in enumerate(lines_after):
                if line.strip() == '' or (line.strip() and line[0].isupper() and i > 3):
                    skill_lines = i
                    break
            insert_position = skills_section_end + len('\n'.join(lines_after[:skill_lines]))
  
        skills_to_add = ", ".join(missing_skills)
        enhanced_text = (
            resume_text[:insert_position] + 
            f", {skills_to_add}" + 
            resume_text[insert_position:]
        )
        return enhanced_text
    else:
        first_section = re.search(r'\n\n[A-Z]', resume_text)
        if first_section:
            insert_position = first_section.start()
            skills_to_add = ", ".join(missing_skills)
            enhanced_text = (
                resume_text[:insert_position] + 
                f"\n\nTECHNICAL SKILLS\n{skills_to_add}\n" + 
                resume_text[insert_position:]
            )
            return enhanced_text
        else:
            skills_to_add = ", ".join(missing_skills)
            return resume_text + f"\n\nTECHNICAL SKILLS\n{skills_to_add}"


def create_pdf_download(text, filename="enhanced_resume.pdf"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    story = []
    
    paragraphs = text.split('\n')
    
    for para in paragraphs:
        if para.strip():
            if para.strip().isupper() or (len(para) < 50 and para.strip()[0].isupper()):
                style = ParagraphStyle(
                    'CustomHeading',
                    parent=styles['Heading2'],
                    fontSize=12,
                    textColor='black',
                    spaceAfter=6,
                )
                story.append(Paragraph(para.strip(), style))
            else:
                story.append(Paragraph(para.strip(), styles['BodyText']))
        else:
            story.append(Spacer(1, 0.2*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    st.title("Resume ATS Score and Enhancement Tool")

    resume = st.file_uploader("Upload your resume (PDF only):", type=["pdf"])
    job_description = st.text_area("Paste the job description here:")

    if st.button("Run ATS"):
        if resume is None:
            st.error("Please upload a resume PDF file.")
            return
        if not job_description.strip():
            st.error("Please provide the job description.")
            return

        st.write(f"Processing resume: {resume.name}")

        resume_text = documentLoader(resume)
       
        resume_docs = textSplitter(resume_text)
        jd_docs = textSplitter(job_description)

        llm = getLLMModel()

        resume_db = getDB(resume_docs)
        jd_db = getDB(jd_docs)

        resume_retriever = resume_db.as_retriever()
        jd_retriever = jd_db.as_retriever()

        resume_qa = RetrievalQA.from_chain_type(llm=llm, retriever=resume_retriever)
        jd_qa = RetrievalQA.from_chain_type(llm=llm, retriever=jd_retriever)

        resume_query = "What are the technical skills listed in this resume?"
        jd_query = "What are the technical skills, languages, and tools listed in this job description?"

        resume_result = resume_qa.invoke({"query": resume_query})
        jd_result = jd_qa.invoke({"query": jd_query})

        ats_score, missing_skills = calculateATS(resume_result, jd_result)

        st.write(f"**ATS Score based on the job description: {ats_score}%**")

        if missing_skills:
            enhanced_resume_text = addMissingSkillsToResume(resume_text, missing_skills)
            
            st.subheader("Enhanced Resume Content Preview:")
            st.text_area("Enhanced Resume Text", enhanced_resume_text, height=300)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Enhanced Resume (TXT)",
                    data=enhanced_resume_text,
                    file_name="enhanced_resume.txt",
                    mime="text/plain"
                )
            
            with col2:
                pdf_buffer = create_pdf_download(enhanced_resume_text)
                st.download_button(
                    label="Download Enhanced Resume (PDF)",
                    data=pdf_buffer,
                    file_name="enhanced_resume.pdf",
                    mime="application/pdf"
                )
        else:
            st.success("Your resume already contains all the required skills from the job description!")
            st.info("No modifications needed.")


if __name__ == "__main__":
    main()
