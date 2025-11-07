from textwrap import dedent
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List, Dict, Type, ClassVar
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables...
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

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
        model="gemma3:1b",
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
            
            return json.dumps({"status": "success", "message": f"Email sent successfully to {to}"})
        except Exception as e:
            return json.dumps({"error": f"Error sending email: {str(e)}"})

class DetailedSalesCrew:
    def __init__(self, target_emails: List[str], use_gpt: bool = False):
        self.target_emails = target_emails
        self.llm = get_llm(use_gpt)
        self.email_tool = EmailSender()
        
    def create_agents(self):
        # Research Agent
        self.researcher = Agent(
            role='Company Research Specialist',
            goal='Analyze companies and gather comprehensive information',
            backstory=dedent("""You are an expert researcher specializing in 
                company analysis. You excel at finding detailed information 
                about companies, their products, and market presence."""),
            tools=[search_tool],
            verbose=True,
            llm=self.llm,
            max_iter=100,
            allow_delegation=False,
            max_rpm=50,
            max_retry_limit=3
        )
        
        # News Agent
        self.news_analyst = Agent(
            role='News and Trends Analyst',
            goal='Find and analyze relevant news and industry trends',
            backstory=dedent("""You are skilled at identifying relevant news 
                and understanding industry trends. You can connect company 
                activities to broader market movements."""),
            tools=[search_tool],
            verbose=True,
            llm=self.llm,
            max_iter=75,
            allow_delegation=False,
            max_rpm=30,
            max_retry_limit=2
        )
        
        # Content Writer
        self.writer = Agent(
            role='Outreach Content Specialist',
            goal='Create highly personalized email content',
            backstory=dedent("""You are an expert at crafting personalized 
                outreach emails that resonate with recipients. You excel at 
                combining company research with industry insights. You are founder of explainx.ai and your name is Yash Thakker, which is what should be mentioned in the email."""),
            tools=[self.email_tool],
            verbose=True,
            llm=self.llm,
            max_iter=50,
            allow_delegation=False,
            max_rpm=20,
            max_retry_limit=2
        )
        
        return [self.researcher, self.news_analyst, self.writer]
    
    def create_tasks(self, email: str):
        # Extract domain from email
        domain = email.split('@')[1]
        company_name = domain.split('.')[0]
        
        # Research Task
        research_task = Task(
            description=dedent(f"""Research {company_name} ({domain}) thoroughly.
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
                {company_name} and their industry.
                
                Step-by-step approach:
                1. Search for company news from the last 3 months
                2. Research industry trends affecting them
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
            description=dedent(f"""Create a personalized email for {email} using 
                the research and news analysis.
                
                Step-by-step approach:
                1. Extract key insights from research
                2. Identify compelling news points
                3. Craft attention-grabbing subject
                4. Write personalized introduction
                5. Present value proposition
                
                Guidelines:
                - Keep subject line engaging but professional
                - Reference specific company details from research
                - Mention relevant news or trends
                - Focus on value proposition
                - Keep email concise (150-200 words)
                - Include clear call to action
                
                Format the response as JSON with 'to', 'subject', and 'body' fields."""),
            agent=self.writer,
            expected_output=dedent("""JSON formatted email content with subject 
                line and body text."""),
            context=[research_task, news_task]
        )
        
        return [research_task, news_task, email_task]
    
    def run(self):
        """Process each email and create personalized outreach"""
        all_results = []
        
        for email in self.target_emails:
            print(f"\nProcessing email: {email}")
            
            # Create crew for this email
            crew = Crew(
                agents=self.create_agents(),
                tasks=self.create_tasks(email),
                process=Process.sequential,
                verbose=True,
                max_rpm=100
            )
            
            # Execute the crew's tasks
            result = crew.kickoff()
            all_results.append({
                "email": email,
                "result": result
            })
        
        return all_results

def main():
    print("\n🔍 Welcome to Sales Outreach Crew!")
    print("\nAvailable Models:")
    print("1. OpenAI GPT-4 Turbo (Requires API key)")
    print("2. Local DeepSeek Coder (Requires Ollama)")
    
    use_gpt = input("\nUse OpenAI GPT-4? (yes/no): ").lower() == 'yes'
    
    if not use_gpt:
        print("\nUsing Ollama with DeepSeek Coder")
        print("Ensure Ollama is running: ollama run deepseek-coder:latest")
    
    target_emails = [
        "pratham@explainx.ai"
    ]
    
    try:
        # Initialize and run the sales crew
        sales_crew = DetailedSalesCrew(target_emails, use_gpt)
        results = sales_crew.run()
        
        # Print results
        for result in results:
            print(f"\nResults for {result['email']}:")
            print(result['result'])
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if use_gpt:
            print("\nTip: Check your OpenAI API key and SERPER_API_KEY")
        else:
            print("\nTip: Ensure Ollama is running with deepseek-coder:latest")
            print("Run: ollama run deepseek-coder:latest")
            print("Also check your SERPER_API_KEY")

if __name__ == "__main__":
    main()