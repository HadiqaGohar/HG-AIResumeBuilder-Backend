import os
import io
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pdfplumber
import mammoth
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from typing import Optional, Dict


# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000", "https://hg-ai-resume-builder.vercel.app"],  # Replace with your Vercel URL
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variable Loading ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- AI Agent Initialization (Gemini via OpenAI SDK compatibility) ---
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Pydantic models for request body
class ResumeInput(BaseModel):
    education: list[str]
    skills: list[str]

class ResumeData(BaseModel):
    name: str = ""
    tag: str = ""
    email: str = ""
    location: str = ""
    number: str = ""
    phone: str = ""
    summary: str = ""
    websites: list[str] = []
    website: str = ""
    linkedin: str = ""
    github: str = ""
    skills: list[str] = []
    education: list[str] = []
    experience: list[str] = []
    student: list[str] = []
    courses: list[str] = []
    internships: list[str] = []
    extracurriculars: list[str] = []
    hobbies: list[str] = []
    references: list[str] = []
    languages: list[str] = []
    awards: list[str] = []
    extra: list[str] = []
    certifications: list = []
    projects: list = []
    headerColor: str = "#a3e4db"
    nameFontStyle: str = "regular"
    nameFontSize: int = 18
    tagFontStyle: str = "regular"
    tagFontSize: int = 14
    summaryFontStyle: str = "regular"
    summaryFontSize: int = 12
    image: str = ""
    profileImage: str = ""

class JobOptimizationInput(BaseModel):
    job_description: str
    resume_data: ResumeData

class SkillSuggestionInput(BaseModel):
    profession: str
    current_skills: list[str] = []




@app.post("/api/resume")
async def generate_resume_summary(input_data: ResumeInput):
    try:
        # Convert lists to comma-separated strings for the prompt
        education_str = ", ".join(input_data.education)
        skills_str = ", ".join(input_data.skills)
        # extra_str = ", ".join(input_data.extra)
        # student_str = ", ".join(input_data.student)
        # experiene_str = ", ".join(input_data.experience)
        # language_str = ", ".join(input_data.language)
        # award_str = ", ".join(input_data.award)

        # Call Gemini model via OpenAI SDK compatibility
        response = await external_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a professional resume summary for a candidate with education: {education_str} and skills: {skills_str}. Keep it concise, professional, and ATS-friendly. Limit to 3-4 sentences."
                }
            ]
        )

        summary = response.choices[0].message.content or ""
        return {"summary": summary}

    except Exception as e:
        print(f"Error generating resume summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.post("/api/resume/extract")
async def extract_resume_data(file: UploadFile = File(...)):
    """Extract structured data from uploaded resume"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    extracted_text = ""
    
    try:
        content = await file.read()
        
        # Extract text based on file type
        if file.filename.lower().endswith(".pdf"):
            import io
            pdf_file = io.BytesIO(content)
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    extracted_text += page_text + "\n"
                    
        elif file.filename.lower().endswith(".docx"):
            import io
            import mammoth
            docx_file = io.BytesIO(content)
            result = mammoth.extract_raw_text(docx_file)
            extracted_text = result.value
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Only PDF and DOCX files are supported."
            )
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file."
            )
        
        # Use Gemini to parse extracted text
        response = await external_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": f"""Parse the following resume text into structured JSON data. Extract these fields:
                    - name (string): Full name
                    - tag (string): Professional title or role  
                    - email (string): Email address
                    - location (string): City, State or address
                    - number (string): Phone number
                    - summary (string): Professional summary
                    - websites (array): URLs like LinkedIn, portfolio
                    - skills (array): Technical and soft skills
                    - education (array): Degrees, schools, years
                    - experience (array): Job titles, companies
                    - student (array): Student status
                    - courses (array): Relevant coursework
                    - internships (array): Internship experiences
                    - extracurriculars (array): Activities, volunteer work
                    - hobbies (array): Personal interests
                    - references (array): Professional references
                    - languages (array): Spoken languages

                    Return ONLY valid JSON without markdown formatting.
                    Use empty string "" for missing fields and empty array [] for missing lists.

                    Resume text:
                    {extracted_text}
                    """
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content or "{}"
        
        # Clean the response
        if result.startswith("```json"):
            result = result[7:-3]
        elif result.startswith("```"):
            result = result[3:-3]
        
        try:
            structured_data = json.loads(result)
        except json.JSONDecodeError:
            # Fallback structure if JSON parsing fails
            structured_data = {
                "name": "",
                "tag": "",
                "email": "",
                "location": "",
                "number": "",
                "summary": "",
                "websites": [],
                "skills": [],
                "education": [],
                "experience": [],
                "student": [],
                "courses": [],
                "internships": [],
                "extracurriculars": [],
                "hobbies": [],
                "references": [],
                "languages": []
            }
        
        return structured_data
        
    except Exception as e:
        print(f"Error extracting resume data: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to extract resume data: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini connection
        test_response = await external_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        gemini_status = "connected"
    except Exception as e:
        print(f"Gemini connection failed: {e}")
        gemini_status = "disconnected"
    
    return {
        "api_status": "healthy",
        "gemini_status": gemini_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/resume/edit")
async def edit_resume_data(resume_data: ResumeData):
    """Save/edit resume data"""
    try:
        # For now, just return the data as confirmation
        # In a real app, you'd save this to a database
        return resume_data.dict()
    except Exception as e:
        print(f"Error editing resume data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to edit resume data: {str(e)}")

@app.post("/api/resume/optimize")
async def optimize_resume(input_data: JobOptimizationInput):
    """Optimize resume for specific job description"""
    try:
        job_desc = input_data.job_description
        resume = input_data.resume_data
        
        # Create optimization prompt
        prompt = f"""
        Analyze this job description and optimize the resume accordingly:
        
        JOB DESCRIPTION:
        {job_desc}
        
        CURRENT RESUME DATA:
        Name: {resume.name}
        Title: {resume.tag}
        Summary: {resume.summary}
        Skills: {', '.join(resume.skills)}
        Experience: {', '.join(resume.experience)}
        Education: {', '.join(resume.education)}
        
        Please provide:
        1. An optimized professional summary that matches the job requirements
        2. 5-8 additional skills that would be relevant for this job
        3. Key keywords from the job description that match the resume
        4. 3-5 specific improvement suggestions
        
        Return the response in this exact JSON format:
        {{
            "optimized_summary": "...",
            "suggested_skills": ["skill1", "skill2", ...],
            "keyword_matches": ["keyword1", "keyword2", ...],
            "improvement_suggestions": ["suggestion1", "suggestion2", ...]
        }}
        """
        
        response = await external_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content or "{}"
        
        # Clean the response
        if result.startswith("```json"):
            result = result[7:-3]
        elif result.startswith("```"):
            result = result[3:-3]
        
        try:
            optimization_data = json.loads(result)
        except json.JSONDecodeError:
            # Fallback response
            optimization_data = {
                "optimized_summary": resume.summary,
                "suggested_skills": [],
                "keyword_matches": [],
                "improvement_suggestions": ["Unable to generate specific suggestions at this time."]
            }
        
        return optimization_data
        
    except Exception as e:
        print(f"Error optimizing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize resume: {str(e)}")

@app.post("/api/resume/skills/suggest")
async def suggest_skills(input_data: SkillSuggestionInput):
    """Suggest relevant skills for a profession"""
    try:
        profession = input_data.profession
        current_skills = input_data.current_skills
        
        prompt = f"""
        Suggest 8-10 relevant skills for someone with the profession: "{profession}"
        
        Current skills they already have: {', '.join(current_skills)}
        
        Provide skills that would complement their existing skillset and are in-demand for this profession.
        Focus on both technical and soft skills that are relevant.
        
        Return only a JSON array of skill names:
        ["skill1", "skill2", "skill3", ...]
        """
        
        response = await external_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content or "[]"
        
        # Clean the response
        if result.startswith("```json"):
            result = result[7:-3]
        elif result.startswith("```"):
            result = result[3:-3]
        
        try:
            suggested_skills = json.loads(result)
            if not isinstance(suggested_skills, list):
                suggested_skills = []
        except json.JSONDecodeError:
            suggested_skills = []
        
        return {"suggested_skills": suggested_skills}
        
    except Exception as e:
        print(f"Error suggesting skills: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to suggest skills: {str(e)}")

@app.post("/api/resume/summary")
async def generate_resume_summary_v2(input_data: ResumeInput):
    """Generate resume summary (alternative endpoint)"""
    return await generate_resume_summary(input_data)

# --- Chatbot Endpoint ---

# Add this Pydantic model near your other models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[dict] = None  # For resume_data

# In-memory chat history store (temporary, replace with database for production)
chat_history: Dict[str, list] = {}

@app.post("/api/chatbot")
async def chatbot(request: ChatRequest):
    
    try:
        # Get session_id or use default
        session_id = request.session_id or "default_session"
        
        # Initialize chat history for session if not exists
        if session_id not in chat_history:
            chat_history[session_id] = []

        # Get resume data from context (if provided)
        resume_data = request.context.get("resume_data", {}) if request.context else {}
        resume_summary = (
            f"Name: {resume_data.get('name', '')}, "
            f"Skills: {', '.join(resume_data.get('skills', []))}, "
            f"Education: {', '.join(resume_data.get('education', []))}"
        ) if resume_data else "No resume data provided."

        # Prepare chat history for prompt
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[session_id]])

        # Construct prompt
        prompt = f"""
You are the HG Resume Builder assistant, created by Hadiqa Gohar. Answer the user's question based on the provided resume data, chat history, and the following context:

Context: HG Resume Builder offers AI-powered resume creation with three featured templates: Chameleon Pro Resume (ATS-friendly, customizable colors), Modern Professional (two-column layout, timeline design), and Creative Sidebar (sidebar design, gradient colors). Users can enhance CVs, export PDFs, and get expert feedback.

Resume Data: {resume_summary}
Chat History: {history_str}
Question: {request.message}
Answer in a concise, professional, and ATS-friendly manner. Provide 2-3 relevant suggestions for follow-up questions.
"""

        # Call Gemini model using the existing external_client
        response = await external_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )

        answer = response.choices[0].message.content or "Sorry, I couldn't generate a response."

        # Update chat history
        chat_history[session_id].append({"role": "user", "content": request.message})
        chat_history[session_id].append({"role": "assistant", "content": answer})

        # Keep history manageable (e.g., last 10 messages)
        if len(chat_history[session_id]) > 10:
            chat_history[session_id] = chat_history[session_id][-10:]

        # Return response compatible with frontend
        return {
            "response": answer,
            "type": "answer",
            "sources": [],  # Add sources if you integrate a knowledge base later
            "suggestions": [
                "How can I improve my resume summary?",
                "What skills should I add?",
                "Show me templates"
            ],
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }

    except Exception as e:
        print(f"Error in chatbot endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chatbot query: {str(e)}")

@app.post("/api/chatbot/session/clear")
async def clear_session(request: ChatRequest):
    try:
        session_id = request.session_id or "default_session"
        if session_id in chat_history:
            del chat_history[session_id]
        return {"message": "Session cleared successfully"}
    except Exception as e:
        print(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")
    

@app.get("/")
async def root():
    
    return {"message": "FastAPI resume backend with Gemini is running."}
