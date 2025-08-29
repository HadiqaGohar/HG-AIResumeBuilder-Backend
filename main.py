# import os
# import io
# import json
# from datetime import datetime
# from fastapi import FastAPI, HTTPException, Request, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import pdfplumber
# import mammoth
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
# from typing import Optional, Dict


# # Load environment variables from .env file
# load_dotenv()

# app = FastAPI()

# # --- CORS Configuration ---

# origins = [
#     "http://localhost:3000",
#     "https://hg-ai-resume-builder.vercel.app",
#     "https://*.vercel.app",
#     "https://*.railway.app",
# ]


# app.add_middleware(
#     CORSMiddleware,
#     # allow_origins=["http://localhost:3000", "https://hg-ai-resume-builder.vercel.app"],  # Replace with your Vercel URL
#     allow_origins=origins,
#     # allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
    
# )

# # --- Environment Variable Loading ---
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # --- AI Agent Initialization (Gemini via OpenAI SDK compatibility) ---
# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model = OpenAIChatCompletionsModel(
#     openai_client=external_client,
#     model="gemini-2.0-flash",
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True,
# )

# # Pydantic models for request body
# class ResumeInput(BaseModel):
#     education: list[str]
#     skills: list[str]

# class ResumeData(BaseModel):
#     name: str = ""
#     tag: str = ""
#     email: str = ""
#     location: str = ""
#     number: str = ""
#     phone: str = ""
#     summary: str = ""
#     websites: list[str] = []
#     website: str = ""
#     linkedin: str = ""
#     github: str = ""
#     skills: list[str] = []
#     education: list[str] = []
#     experience: list[str] = []
#     student: list[str] = []
#     courses: list[str] = []
#     internships: list[str] = []
#     extracurriculars: list[str] = []
#     hobbies: list[str] = []
#     references: list[str] = []
#     languages: list[str] = []
#     awards: list[str] = []
#     extra: list[str] = []
#     certifications: list = []
#     projects: list = []
#     headerColor: str = "#a3e4db"
#     nameFontStyle: str = "regular"
#     nameFontSize: int = 18
#     tagFontStyle: str = "regular"
#     tagFontSize: int = 14
#     summaryFontStyle: str = "regular"
#     summaryFontSize: int = 12
#     image: str = ""
#     profileImage: str = ""

# class JobOptimizationInput(BaseModel):
#     job_description: str
#     resume_data: ResumeData

# class SkillSuggestionInput(BaseModel):
#     profession: str
#     current_skills: list[str] = []




# @app.post("/api/resume")
# async def generate_resume_summary(input_data: ResumeInput):
#     try:
#         # Convert lists to comma-separated strings for the prompt
#         education_str = ", ".join(input_data.education)
#         skills_str = ", ".join(input_data.skills)
#         # extra_str = ", ".join(input_data.extra)
#         # student_str = ", ".join(input_data.student)
#         # experiene_str = ", ".join(input_data.experience)
#         # language_str = ", ".join(input_data.language)
#         # award_str = ", ".join(input_data.award)

#         # Call Gemini model via OpenAI SDK compatibility
#         response = await external_client.chat.completions.create(
#             model="gemini-2.0-flash",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"Generate a professional resume summary for a candidate with education: {education_str} and skills: {skills_str}. Keep it concise, professional, and ATS-friendly. Limit to 3-4 sentences."
#                 }
#             ]
#         )

#         summary = response.choices[0].message.content or ""
#         return {"summary": summary}

#     except Exception as e:
#         print(f"Error generating resume summary: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

# @app.post("/api/resume/extract")
# async def extract_resume_data(file: UploadFile = File(...)):
#     """Extract structured data from uploaded resume"""
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="No file provided")
    
#     extracted_text = ""
    
#     try:
#         content = await file.read()
        
#         # Extract text based on file type
#         if file.filename.lower().endswith(".pdf"):
#             import io
#             pdf_file = io.BytesIO(content)
#             with pdfplumber.open(pdf_file) as pdf:
#                 for page in pdf.pages:
#                     page_text = page.extract_text() or ""
#                     extracted_text += page_text + "\n"
                    
#         elif file.filename.lower().endswith(".docx"):
#             import io
#             import mammoth
#             docx_file = io.BytesIO(content)
#             result = mammoth.extract_raw_text(docx_file)
#             extracted_text = result.value
#         else:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Unsupported file format. Only PDF and DOCX files are supported."
#             )
        
#         if not extracted_text.strip():
#             raise HTTPException(
#                 status_code=400,
#                 detail="No text could be extracted from the file."
#             )
        
#         # Use Gemini to parse extracted text
#         response = await external_client.chat.completions.create(
#             model="gemini-2.0-flash",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"""Parse the following resume text into structured JSON data. Extract these fields:
#                     - name (string): Full name
#                     - tag (string): Professional title or role  
#                     - email (string): Email address
#                     - location (string): City, State or address
#                     - number (string): Phone number
#                     - summary (string): Professional summary
#                     - websites (array): URLs like LinkedIn, portfolio
#                     - skills (array): Technical and soft skills
#                     - education (array): Degrees, schools, years
#                     - experience (array): Job titles, companies
#                     - student (array): Student status
#                     - courses (array): Relevant coursework
#                     - internships (array): Internship experiences
#                     - extracurriculars (array): Activities, volunteer work
#                     - hobbies (array): Personal interests
#                     - references (array): Professional references
#                     - languages (array): Spoken languages

#                     Return ONLY valid JSON without markdown formatting.
#                     Use empty string "" for missing fields and empty array [] for missing lists.

#                     Resume text:
#                     {extracted_text}
#                     """
#                 }
#             ],
#             max_tokens=1500,
#             temperature=0.3
#         )
        
#         result = response.choices[0].message.content or "{}"
        
#         # Clean the response
#         if result.startswith("```json"):
#             result = result[7:-3]
#         elif result.startswith("```"):
#             result = result[3:-3]
        
#         try:
#             structured_data = json.loads(result)
#         except json.JSONDecodeError:
#             # Fallback structure if JSON parsing fails
#             structured_data = {
#                 "name": "",
#                 "tag": "",
#                 "email": "",
#                 "location": "",
#                 "number": "",
#                 "summary": "",
#                 "websites": [],
#                 "skills": [],
#                 "education": [],
#                 "experience": [],
#                 "student": [],
#                 "courses": [],
#                 "internships": [],
#                 "extracurriculars": [],
#                 "hobbies": [],
#                 "references": [],
#                 "languages": []
#             }
        
#         return structured_data
        
#     except Exception as e:
#         print(f"Error extracting resume data: {str(e)}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Failed to extract resume data: {str(e)}"
#         )

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     try:
#         # Test Gemini connection
#         test_response = await external_client.chat.completions.create(
#             model="gemini-2.0-flash",
#             messages=[{"role": "user", "content": "Hello"}],
#             max_tokens=10
#         )
#         gemini_status = "connected"
#     except Exception as e:
#         print(f"Gemini connection failed: {e}")
#         gemini_status = "disconnected"
    
#     return {
#         "api_status": "healthy",
#         "gemini_status": gemini_status,
#         "timestamp": datetime.now().isoformat()
#     }

# @app.post("/api/resume/edit")
# async def edit_resume_data(resume_data: ResumeData):
#     """Save/edit resume data"""
#     try:
#         # For now, just return the data as confirmation
#         # In a real app, you'd save this to a database
#         return resume_data.dict()
#     except Exception as e:
#         print(f"Error editing resume data: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to edit resume data: {str(e)}")

# @app.post("/api/resume/optimize")
# async def optimize_resume(input_data: JobOptimizationInput):
#     """Optimize resume for specific job description"""
#     try:
#         job_desc = input_data.job_description
#         resume = input_data.resume_data
        
#         # Create optimization prompt
#         prompt = f"""
#         Analyze this job description and optimize the resume accordingly:
        
#         JOB DESCRIPTION:
#         {job_desc}
        
#         CURRENT RESUME DATA:
#         Name: {resume.name}
#         Title: {resume.tag}
#         Summary: {resume.summary}
#         Skills: {', '.join(resume.skills)}
#         Experience: {', '.join(resume.experience)}
#         Education: {', '.join(resume.education)}
        
#         Please provide:
#         1. An optimized professional summary that matches the job requirements
#         2. 5-8 additional skills that would be relevant for this job
#         3. Key keywords from the job description that match the resume
#         4. 3-5 specific improvement suggestions
        
#         Return the response in this exact JSON format:
#         {{
#             "optimized_summary": "...",
#             "suggested_skills": ["skill1", "skill2", ...],
#             "keyword_matches": ["keyword1", "keyword2", ...],
#             "improvement_suggestions": ["suggestion1", "suggestion2", ...]
#         }}
#         """
        
#         response = await external_client.chat.completions.create(
#             model="gemini-2.0-flash",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=1500,
#             temperature=0.3
#         )
        
#         result = response.choices[0].message.content or "{}"
        
#         # Clean the response
#         if result.startswith("```json"):
#             result = result[7:-3]
#         elif result.startswith("```"):
#             result = result[3:-3]
        
#         try:
#             optimization_data = json.loads(result)
#         except json.JSONDecodeError:
#             # Fallback response
#             optimization_data = {
#                 "optimized_summary": resume.summary,
#                 "suggested_skills": [],
#                 "keyword_matches": [],
#                 "improvement_suggestions": ["Unable to generate specific suggestions at this time."]
#             }
        
#         return optimization_data
        
#     except Exception as e:
#         print(f"Error optimizing resume: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to optimize resume: {str(e)}")

# @app.post("/api/resume/skills/suggest")
# async def suggest_skills(input_data: SkillSuggestionInput):
#     """Suggest relevant skills for a profession"""
#     try:
#         profession = input_data.profession
#         current_skills = input_data.current_skills
        
#         prompt = f"""
#         Suggest 8-10 relevant skills for someone with the profession: "{profession}"
        
#         Current skills they already have: {', '.join(current_skills)}
        
#         Provide skills that would complement their existing skillset and are in-demand for this profession.
#         Focus on both technical and soft skills that are relevant.
        
#         Return only a JSON array of skill names:
#         ["skill1", "skill2", "skill3", ...]
#         """
        
#         response = await external_client.chat.completions.create(
#             model="gemini-2.0-flash",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=500,
#             temperature=0.3
#         )
        
#         result = response.choices[0].message.content or "[]"
        
#         # Clean the response
#         if result.startswith("```json"):
#             result = result[7:-3]
#         elif result.startswith("```"):
#             result = result[3:-3]
        
#         try:
#             suggested_skills = json.loads(result)
#             if not isinstance(suggested_skills, list):
#                 suggested_skills = []
#         except json.JSONDecodeError:
#             suggested_skills = []
        
#         return {"suggested_skills": suggested_skills}
        
#     except Exception as e:
#         print(f"Error suggesting skills: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to suggest skills: {str(e)}")

# @app.post("/api/resume/summary")
# async def generate_resume_summary_v2(input_data: ResumeInput):
#     """Generate resume summary (alternative endpoint)"""
#     return await generate_resume_summary(input_data)

# # --- Chatbot Endpoint ---

# # Add this Pydantic model near your other models
# class ChatRequest(BaseModel):
#     message: str
#     session_id: Optional[str] = None
#     context: Optional[dict] = None  # For resume_data

# # In-memory chat history store (temporary, replace with database for production)
# chat_history: Dict[str, list] = {}

# @app.post("/api/chatbot")
# async def chatbot(request: ChatRequest):
    
#     try:
#         # Get session_id or use default
#         session_id = request.session_id or "default_session"
        
#         # Initialize chat history for session if not exists
#         if session_id not in chat_history:
#             chat_history[session_id] = []

#         # Get resume data from context (if provided)
#         resume_data = request.context.get("resume_data", {}) if request.context else {}
#         resume_summary = (
#             f"Name: {resume_data.get('name', '')}, "
#             f"Skills: {', '.join(resume_data.get('skills', []))}, "
#             f"Education: {', '.join(resume_data.get('education', []))}"
#         ) if resume_data else "No resume data provided."

#         # Prepare chat history for prompt
#         history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[session_id]])

#         # Construct prompt
#         prompt = f"""
# You are the HG Resume Builder assistant, created by Hadiqa Gohar. Answer the user's question based on the provided resume data, chat history, and the following context:

# Context: HG Resume Builder offers AI-powered resume creation with three featured templates: Chameleon Pro Resume (ATS-friendly, customizable colors), Modern Professional (two-column layout, timeline design), and Creative Sidebar (sidebar design, gradient colors). Users can enhance CVs, export PDFs, and get expert feedback.

# Resume Data: {resume_summary}
# Chat History: {history_str}
# Question: {request.message}
# Answer in a concise, professional, and ATS-friendly manner. Provide 2-3 relevant suggestions for follow-up questions.
# """

#         # Call Gemini model using the existing external_client
#         response = await external_client.chat.completions.create(
#             model="gemini-2.0-flash",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=500,
#             temperature=0.3
#         )

#         answer = response.choices[0].message.content or "Sorry, I couldn't generate a response."

#         # Update chat history
#         chat_history[session_id].append({"role": "user", "content": request.message})
#         chat_history[session_id].append({"role": "assistant", "content": answer})

#         # Keep history manageable (e.g., last 10 messages)
#         if len(chat_history[session_id]) > 10:
#             chat_history[session_id] = chat_history[session_id][-10:]

#         # Return response compatible with frontend
#         return {
#             "response": answer,
#             "type": "answer",
#             "sources": [],  # Add sources if you integrate a knowledge base later
#             "suggestions": [
#                 "How can I improve my resume summary?",
#                 "What skills should I add?",
#                 "Show me templates"
#             ],
#             "timestamp": datetime.now().isoformat(),
#             "session_id": session_id
#         }

#     except Exception as e:
#         print(f"Error in chatbot endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to process chatbot query: {str(e)}")

# @app.post("/api/chatbot/session/clear")
# async def clear_session(request: ChatRequest):
#     try:
#         session_id = request.session_id or "default_session"
#         if session_id in chat_history:
#             del chat_history[session_id]
#         return {"message": "Session cleared successfully"}
#     except Exception as e:
#         print(f"Error clearing session: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")
    

# @app.get("/")
# async def root():
    
#     return {"message": "FastAPI resume backend with Gemini is running."}

# ----------------------------------------------------------------------------
import os
import io
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

import pdfplumber
import mammoth
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log") if os.environ.get("RAILWAY_ENVIRONMENT") != "production" else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables for application state
app_state = {
    "startup_time": datetime.now(),
    "healthy": True,
    "gemini_connected": False,
    "total_requests": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up")
    
    # Test Gemini connection on startup
    try:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            app_state["gemini_connected"] = False
        else:
            # Test connection
            test_client = AsyncOpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            test_response = await test_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=5
            )
            app_state["gemini_connected"] = True
            logger.info("Gemini API connection successful")
    except Exception as e:
        logger.error(f"Gemini API connection failed: {str(e)}")
        app_state["gemini_connected"] = False
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")
    app_state["healthy"] = False

app = FastAPI(
    title="HG AI Resume Builder API",
    description="AI-powered resume building and optimization service",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://hg-ai-resume-builder.vercel.app",
    "https://*.vercel.app",
    "https://*.railway.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency for API Key Validation ---
async def verify_api_key():
    """Dependency to verify API key is present"""
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY not configured"
        )

# --- AI Client Initialization ---
def get_gemini_client():
    """Get Gemini client with error handling"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable is not set")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI service configuration error"
        )
    
    try:
        client = AsyncOpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=30.0  # 30 second timeout
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI service initialization failed: {str(e)}"
        )

# --- Pydantic Models ---
class ResumeInput(BaseModel):
    education: List[str] = Field(..., description="List of educational qualifications")
    skills: List[str] = Field(..., description="List of skills")

class ResumeData(BaseModel):
    name: str = Field("", description="Full name")
    tag: str = Field("", description="Professional title or tagline")
    email: str = Field("", description="Email address")
    location: str = Field("", description="Location/City")
    number: str = Field("", description="Phone number")
    phone: str = Field("", description="Alternative phone number")
    summary: str = Field("", description="Professional summary")
    websites: List[str] = Field([], description="List of websites/URLs")
    website: str = Field("", description="Primary website")
    linkedin: str = Field("", description="LinkedIn profile URL")
    github: str = Field("", description="GitHub profile URL")
    skills: List[str] = Field([], description="List of skills")
    education: List[str] = Field([], description="List of education details")
    experience: List[str] = Field([], description="List of work experiences")
    student: List[str] = Field([], description="Student information")
    courses: List[str] = Field([], description="List of courses")
    internships: List[str] = Field([], description="List of internships")
    extracurriculars: List[str] = Field([], description="Extracurricular activities")
    hobbies: List[str] = Field([], description="List of hobbies")
    references: List[str] = Field([], description="List of references")
    languages: List[str] = Field([], description="List of languages spoken")
    awards: List[str] = Field([], description="List of awards")
    extra: List[str] = Field([], description="Additional information")
    certifications: List[str] = Field([], description="List of certifications")
    projects: List[str] = Field([], description="List of projects")
    headerColor: str = Field("#a3e4db", description="Header color for resume")
    nameFontStyle: str = Field("regular", description="Font style for name")
    nameFontSize: int = Field(18, description="Font size for name")
    tagFontStyle: str = Field("regular", description="Font style for tag")
    tagFontSize: int = Field(14, description="Font size for tag")
    summaryFontStyle: str = Field("regular", description="Font style for summary")
    summaryFontSize: int = Field(12, description="Font size for summary")
    image: str = Field("", description="Image URL")
    profileImage: str = Field("", description="Profile image URL")

class JobOptimizationInput(BaseModel):
    job_description: str = Field(..., description="Job description to optimize for")
    resume_data: ResumeData = Field(..., description="Resume data to optimize")

class SkillSuggestionInput(BaseModel):
    profession: str = Field(..., description="Target profession")
    current_skills: List[str] = Field([], description="Current skills list")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Chat message")
    session_id: Optional[str] = Field(None, description="Session ID for chat history")
    context: Optional[Dict[str, Any]] = Field(None, description="Context data including resume")

# --- Utility Functions ---
async def call_gemini_api(messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """Wrapper function to call Gemini API with error handling"""
    try:
        client = get_gemini_client()
        response = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Gemini API call failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service temporarily unavailable: {str(e)}"
        )

def clean_json_response(response_text: str) -> str:
    """Clean JSON response from AI model"""
    if not response_text:
        return "{}"
    
    # Remove markdown code blocks
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    return response_text.strip()

def safe_json_parse(json_text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback"""
    try:
        return json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {json_text}")
        return default

# --- In-memory storage (replace with database in production) ---
chat_history: Dict[str, List[Dict[str, str]]] = {}

# --- API Routes ---
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HG AI Resume Builder API is running",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    app_state["total_requests"] += 1
    
    # Test Gemini connection
    gemini_status = "connected"
    try:
        test_client = get_gemini_client()
        test_response = await test_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Health check"}],
            max_tokens=5
        )
    except Exception as e:
        gemini_status = f"disconnected: {str(e)}"
        logger.warning(f"Health check - Gemini connection failed: {e}")
    
    return {
        "status": "healthy" if app_state["healthy"] else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - app_state["startup_time"]),
        "gemini_status": gemini_status,
        "environment": os.environ.get("RAILWAY_ENVIRONMENT", "development"),
        "port": os.environ.get("PORT", "8000"),
        "total_requests": app_state["total_requests"]
    }

@app.post("/api/resume", tags=["Resume"])
async def generate_resume_summary(
    input_data: ResumeInput,
    api_key_check: None = Depends(verify_api_key)
):
    """Generate a professional resume summary"""
    app_state["total_requests"] += 1
    logger.info(f"Generating resume summary for education: {len(input_data.education)} items, skills: {len(input_data.skills)} items")
    
    try:
        education_str = ", ".join(input_data.education)
        skills_str = ", ".join(input_data.skills)

        prompt = f"Generate a professional resume summary for a candidate with education: {education_str} and skills: {skills_str}. Keep it concise, professional, and ATS-friendly. Limit to 3-4 sentences."
        
        summary = await call_gemini_api([
            {"role": "user", "content": prompt}
        ])
        
        logger.info("Resume summary generated successfully")
        return {"summary": summary}

    except Exception as e:
        logger.error(f"Error generating resume summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to generate summary: {str(e)}"
        )

@app.post("/api/resume/extract", tags=["Resume"])
async def extract_resume_data(
    file: UploadFile = File(...),
    api_key_check: None = Depends(verify_api_key)
):
    """Extract structured data from uploaded resume file"""
    app_state["total_requests"] += 1
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file provided"
        )
    
    logger.info(f"Extracting resume data from file: {file.filename}")
    
    try:
        content = await file.read()
        extracted_text = ""
        
        # Extract text based on file type
        if file.filename.lower().endswith(".pdf"):
            try:
                pdf_file = io.BytesIO(content)
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        extracted_text += page_text + "\n"
            except Exception as e:
                logger.error(f"PDF extraction failed: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to extract text from PDF: {str(e)}"
                )
                
        elif file.filename.lower().endswith(".docx"):
            try:
                docx_file = io.BytesIO(content)
                result = mammoth.extract_raw_text(docx_file)
                extracted_text = result.value
            except Exception as e:
                logger.error(f"DOCX extraction failed: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to extract text from DOCX: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Only PDF and DOCX files are supported."
            )
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the file. Please ensure the file contains text and is not image-based."
            )
        
        # Use Gemini to parse extracted text
        prompt = f"""Parse the following resume text into structured JSON data. Extract these fields:
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
        
        result = await call_gemini_api([
            {"role": "user", "content": prompt}
        ], max_tokens=1500)
        
        cleaned_result = clean_json_response(result)
        structured_data = safe_json_parse(cleaned_result, {
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
        })
        
        logger.info("Resume data extracted successfully")
        return structured_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting resume data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to extract resume data: {str(e)}"
        )

@app.post("/api/resume/edit", tags=["Resume"])
async def edit_resume_data(
    resume_data: ResumeData,
    api_key_check: None = Depends(verify_api_key)
):
    """Save/edit resume data"""
    app_state["total_requests"] += 1
    logger.info("Editing resume data")
    
    try:
        # In a real app, you'd save this to a database
        # For now, return the data as confirmation
        return resume_data.dict()
    except Exception as e:
        logger.error(f"Error editing resume data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to edit resume data: {str(e)}"
        )

@app.post("/api/resume/optimize", tags=["Resume"])
async def optimize_resume(
    input_data: JobOptimizationInput,
    api_key_check: None = Depends(verify_api_key)
):
    """Optimize resume for specific job description"""
    app_state["total_requests"] += 1
    logger.info("Optimizing resume for job description")
    
    try:
        job_desc = input_data.job_description
        resume = input_data.resume_data
        
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
        
        result = await call_gemini_api([
            {"role": "user", "content": prompt}
        ], max_tokens=1500)
        
        cleaned_result = clean_json_response(result)
        optimization_data = safe_json_parse(cleaned_result, {
            "optimized_summary": resume.summary,
            "suggested_skills": [],
            "keyword_matches": [],
            "improvement_suggestions": ["Unable to generate specific suggestions at this time."]
        })
        
        logger.info("Resume optimization completed successfully")
        return optimization_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to optimize resume: {str(e)}"
        )

@app.post("/api/resume/skills/suggest", tags=["Resume"])
async def suggest_skills(
    input_data: SkillSuggestionInput,
    api_key_check: None = Depends(verify_api_key)
):
    """Suggest relevant skills for a profession"""
    app_state["total_requests"] += 1
    logger.info(f"Suggesting skills for profession: {input_data.profession}")
    
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
        
        result = await call_gemini_api([
            {"role": "user", "content": prompt}
        ], max_tokens=500)
        
        cleaned_result = clean_json_response(result)
        suggested_skills = safe_json_parse(cleaned_result, [])
        
        if not isinstance(suggested_skills, list):
            suggested_skills = []
        
        logger.info(f"Suggested {len(suggested_skills)} skills successfully")
        return {"suggested_skills": suggested_skills}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suggesting skills: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to suggest skills: {str(e)}"
        )

@app.post("/api/chatbot", tags=["Chatbot"])
async def chatbot(
    request: ChatRequest,
    api_key_check: None = Depends(verify_api_key)
):
    """Chatbot endpoint for resume-related queries"""
    app_state["total_requests"] += 1
    logger.info(f"Chatbot request for session: {request.session_id or 'default'}")
    
    try:
        session_id = request.session_id or "default_session"
        
        # Initialize chat history for session if not exists
        if session_id not in chat_history:
            chat_history[session_id] = []
            logger.debug(f"Created new chat session: {session_id}")

        # Get resume data from context
        resume_data = request.context.get("resume_data", {}) if request.context else {}
        resume_summary = (
            f"Name: {resume_data.get('name', '')}, "
            f"Skills: {', '.join(resume_data.get('skills', []))}, "
            f"Education: {', '.join(resume_data.get('education', []))}"
        ) if resume_data else "No resume data provided."

        # Prepare chat history for prompt
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[session_id][-5:]])  # Last 5 messages

        prompt = f"""
You are the HG Resume Builder assistant, created by Hadiqa Gohar. Answer the user's question based on the provided resume data, chat history, and the following context:

Context: HG Resume Builder offers AI-powered resume creation with three featured templates: Chameleon Pro Resume (ATS-friendly, customizable colors), Modern Professional (two-column layout, timeline design), and Creative Sidebar (sidebar design, gradient colors). Users can enhance CVs, export PDFs, and get expert feedback.

Resume Data: {resume_summary}
Chat History: {history_str}
Question: {request.message}
Answer in a concise, professional, and ATS-friendly manner. Provide 2-3 relevant suggestions for follow-up questions.
"""

        answer = await call_gemini_api([
            {"role": "user", "content": prompt}
        ], max_tokens=500)

        # Update chat history
        chat_history[session_id].append({"role": "user", "content": request.message})
        chat_history[session_id].append({"role": "assistant", "content": answer})

        # Keep history manageable (last 20 messages)
        if len(chat_history[session_id]) > 20:
            chat_history[session_id] = chat_history[session_id][-20:]

        logger.info(f"Chatbot response generated for session: {session_id}")
        return {
            "response": answer,
            "type": "answer",
            "sources": [],
            "suggestions": [
                "How can I improve my resume summary?",
                "What skills should I add?",
                "Show me templates"
            ],
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chatbot endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to process chatbot query: {str(e)}"
        )

@app.post("/api/chatbot/session/clear", tags=["Chatbot"])
async def clear_session(request: ChatRequest):
    """Clear chat session history"""
    app_state["total_requests"] += 1
    
    try:
        session_id = request.session_id or "default_session"
        if session_id in chat_history:
            del chat_history[session_id]
            logger.info(f"Cleared chat session: {session_id}")
        return {"message": "Session cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to clear session: {str(e)}"
        )

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler"""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# --- Railway-specific configuration ---
# Get port from Railway environment or default to 8000
PORT = int(os.environ.get("PORT", 8000))

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )
