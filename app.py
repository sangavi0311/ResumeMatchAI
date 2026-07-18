
import pdfplumber
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# ---------- Extract Text From PDF ----------
def extract_text(file):
    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


@app.route("/", methods=["GET", "POST"])
def home():

    score = 0
    resume_text = ""
    job_description = ""

    matched = []
    missing = []
    recommendation = ""

    skills = [
        "python",
        "java",
        "sql",
        "html",
        "css",
        "javascript",
        "flask",
        "django",
        "react",
        "node",
        "mongodb",
        "git",
        "machine learning",
        "data analysis",
        "communication",
        "problem solving",
        "teamwork",
        "leadership",
        "excel",
        "power bi",
        "aws"
    ]

    if request.method == "POST":

        resume = request.files.get("resume")
        job_description = request.form.get("job_description")

        if resume and resume.filename and job_description.strip():

            # Extract Resume Text
            resume_text = extract_text(resume)

            print("=========== RESUME TEXT ===========")
            print(resume_text)
            print("===================================")

            # -------- Similarity Score --------
            documents = [resume_text, job_description]

            vectorizer = TfidfVectorizer()

            vectors = vectorizer.fit_transform(documents)

            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            score = round(similarity * 100, 2)

            # -------- Skill Matching --------
            resume_lower = resume_text.lower()
            job_lower = job_description.lower()

            for skill in skills:

                if skill in job_lower:

                    if skill in resume_lower:
                        matched.append(skill.title())
                    else:
                        missing.append(skill.title())

            # -------- Recommendation --------
            if score >= 80:
                recommendation = (
                    "Excellent match! Your resume strongly aligns with the job description."
                )

            elif score >= 60:
                recommendation = (
                    "Good match. Add the missing skills to further improve your resume."
                )

            elif score >= 40:
                recommendation = (
                    "Average match. Include more relevant technical skills and projects."
                )

            else:
                recommendation = (
                    "Low match. Add relevant skills, certifications, projects, and keywords related to the job."
                )

    return render_template(
        "index.html",
        score=score,
        matched=matched,
        missing=missing,
        recommendation=recommendation,
        resume_text=resume_text,
        job_description=job_description
    )



   if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)