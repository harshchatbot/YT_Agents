from textwrap import dedent
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List, Dict, Type, ClassVar
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
import os
import json


# Initialize SerperDev tool
search_tool = SerperDevTool()

def get_llm(use_gpt=False):
    """Get the specified language model"""
    if use_gpt:
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7
        )
    return Ollama(
        model="deepseek-r1:latest",
        base_url="http://localhost:11434",
        temperature=0.7
    )

class EmailInput(BaseModel):
    """Input schema for Email Tool"""
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")

class EmailSender(BaseTool):
    name: str = "Email Sender"
    description: str = "Sends personalized emails using Gmail SMTP"
    args_schema: Type[BaseModel] = EmailInput
    
    smtp_settings: ClassVar[Dict[str, str | int]] = {
        'server': "smtp.gmail.com",
        'port': 587,
        'username': os.getenv('GMAIL_USER'),
        'password': os.getenv('GMAIL_APP_PASSWORD')
    }

    def _run(self, to: str, subject: str, body: str) -> str:
        if not self.smtp_settings['username'] or not self.smtp_settings['password']:
            return json.dumps({"error": "GMAIL_USER and GMAIL_APP_PASSWORD environment variables are required"})
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_settings['username']
            msg['To'] = to
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_settings['server'], self.smtp_settings['port']) as server:
                server.starttls()
                server.login(self.smtp_settings['username'], self.smtp_settings['password'])
                server.send_message(msg)
            
            return json.dumps({
                "status": "success",
                "message": f"Email sent successfully to {to}",
                "to": to,
                "subject": subject,
                "body": body
            })
        except Exception as e:
            return json.dumps({
                "error": f"Error sending email: {str(e)}",
                "to": to,
                "subject": subject,
                "body": body
            })

def load_email_template(industry: str) -> str:
    """Load email template for the given industry"""
    try:
        with open(f'email_templates/{industry.lower()}.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

class DetailedSalesCrew:
    def __init__(self, target_emails: List[dict], use_gpt: bool = False):
        """
        Initialize with list of dicts containing email and industry
        Example: [{"email": "user@domain.com", "industry": "Technology"}]
        """
        # Validate email format
        for email_data in target_emails:
            if not isinstance(email_data, dict) or "email" not in email_data or "industry" not in email_data:
                raise ValueError("Each target email must be a dictionary with 'email' and 'industry' keys")
        
        self.target_emails = target_emails
        self.llm = get_llm(use_gpt)
        self.email_tool = EmailSender()
        
    def create_agents(self, industry: str):
        # Research Agent
        self.researcher = Agent(
            role='Company Research Specialist',
            goal='Analyze companies and gather comprehensive information',
            backstory=dedent(f"""You are an expert researcher specializing in 
                {industry} company analysis. You excel at finding detailed information 
                about companies, their products, and market presence."""),
            tools=[search_tool],
            verbose=True,
            llm=self.llm,
            max_iter=100,
            allow_delegation=False
        )
        
        # News Agent
        self.news_analyst = Agent(
            role='News and Trends Analyst',
            goal='Find and analyze relevant news and industry trends',
            backstory=dedent(f"""You are skilled at identifying relevant news 
                and understanding {industry} industry trends. You can connect company 
                activities to broader market movements."""),
            tools=[search_tool],
            verbose=True,
            llm=self.llm,
            max_iter=75,
            allow_delegation=False
        )
        
        # Content Writer
        template = load_email_template(industry)
        template_context = f"Use this template if available: {template}" if template else ""
        
        self.writer = Agent(
            role='Outreach Content Specialist',
            goal='Create highly personalized email content',
            backstory=dedent(f"""You are an expert at crafting personalized 
                outreach emails for {industry} companies that resonate with recipients. 
                You excel at combining company research with industry insights.
                You are founder of explainx.ai and your name is Yash Thakker, which 
                is what should be mentioned in the email. {template_context}"""),
            tools=[self.email_tool],
            verbose=True,
            llm=self.llm,
            max_iter=50,
            allow_delegation=False
        )
        
        return [self.researcher, self.news_analyst, self.writer]
    
    def create_tasks(self, email: str, industry: str):
        # Extract domain from email
        domain = email.split('@')[1]
        company_name = domain.split('.')[0]
        
        # Research Task
        research_task = Task(
            description=dedent(f"""Research {company_name} ({domain}) thoroughly.
                Consider their position in the {industry} industry.
                
                Step-by-step approach:
                1. Search for company overview and background
                2. Research their products/services in detail
                3. Find information about their team and leadership
                4. Analyze their market position
                5. Identify their tech stack and tools
                
                Focus on:
                - Company's main products/services
                - Value proposition
                - Target market
                - Team information
                - Recent updates or changes
                - Technology stack or tools mentioned
                
                Create a comprehensive profile of the company."""),
            agent=self.researcher,
            expected_output=dedent("""Detailed company profile including all 
                discovered information in a structured format.""")
        )
        
        # News Analysis Task
        news_task = Task(
            description=dedent(f"""Research recent news and developments about 
                {company_name} and their position in the {industry} industry.
                
                Step-by-step approach:
                1. Search for company news from the last 3 months
                2. Research {industry} industry trends affecting them
                3. Analyze competitor movements
                4. Identify market opportunities
                5. Find any company milestones or achievements
                
                Focus on:
                - Recent company news and press releases
                - Industry trends and developments
                - Competitive landscape
                - Market opportunities and challenges
                - Recent achievements or notable events"""),
            agent=self.news_analyst,
            expected_output=dedent("""Comprehensive news analysis including 
                company-specific news and relevant industry trends.""")
        )
        
        # Email Creation Task
        email_task = Task(
            description=dedent(f"""Create and send a personalized email to {email} using 
                the research and news analysis. Consider their position in the 
                {industry} industry.
                
                Step-by-step approach:
                1. Extract key insights from research
                2. Identify compelling news points
                3. Craft attention-grabbing subject
                4. Write personalized introduction
                5. Present value proposition
                
                Guidelines:
                - Keep subject line engaging but professional
                - Reference specific company details from research
                - Mention relevant {industry} trends
                - Focus on value proposition
                - Keep email concise (150-200 words)
                - Include clear call to action
                - Sign as Yash Thakker, Founder at ExplainX.ai
                
                Use the email tool to send the email directly."""),
            agent=self.writer,
            context=[research_task, news_task],
            expected_output=dedent("""Email sent successfully with response from 
                email tool in JSON format.""")
        )
        
        return [research_task, news_task, email_task]
    
    def run(self):
        """Process each email and create personalized outreach"""
        all_results = []
        
        for target in self.target_emails:
            email = target["email"]
            industry = target["industry"]
            
            print(f"\nProcessing email: {email} (Industry: {industry})")
            
            # Create crew for this email
            crew = Crew(
                agents=self.create_agents(industry),
                tasks=self.create_tasks(email, industry),
                process=Process.sequential,
                verbose=True,
                max_rpm=100
            )
            
            # Execute the crew's tasks
            result = crew.kickoff()
            all_results.append({
                "email": email,
                "industry": industry,
                "result": result
            })
        
        return all_results