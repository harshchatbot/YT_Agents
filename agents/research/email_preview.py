import streamlit as st
import pandas as pd
from typing import Dict, List
import json
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import requests
from bs4 import BeautifulSoup
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Type, ClassVar

# Load environment variables
load_dotenv()

# Initialize SerperDev tool
search_tool = SerperDevTool()

# Initialize ScrapeWebsite tool
scrape_tool = ScrapeWebsiteTool()

# Page config
st.set_page_config(
    page_title="Email Preview & Send",
    page_icon="📧",
    layout="wide"
)

# Initialize session state
if 'email_data' not in st.session_state:
    st.session_state.email_data = None
if 'research_data' not in st.session_state:
    st.session_state.research_data = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
if 'website_data' not in st.session_state:
    st.session_state.website_data = None
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None
if 'draft_subject' not in st.session_state:
    st.session_state.draft_subject = ""
if 'draft_body' not in st.session_state:
    st.session_state.draft_body = ""
if 'draft_cc' not in st.session_state:
    st.session_state.draft_cc = "yash@explainx.ai"
if 'draft_reply_to' not in st.session_state:
    st.session_state.draft_reply_to = "yash@explainx.ai"
if 'email_sent' not in st.session_state:
    st.session_state.email_sent = False

def get_llm(use_gpt=True):
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

def validate_email(email: str) -> bool:
    """Simple email validation"""
    return '@' in email and '.' in email.split('@')[1]

def load_email_template(industry: str) -> str:
    """Load email template for given industry"""
    try:
        with open(f'email_templates/{industry.lower()}.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def send_email(to: str, subject: str, body: str, cc: str = None, reply_to: str = None) -> Dict:
    """Send email using Gmail SMTP"""
    smtp_settings = {
        'server': "smtp.gmail.com",
        'port': 587,
        'username': os.getenv('GMAIL_USER'),
        'password': os.getenv('GMAIL_APP_PASSWORD')
    }
    
    if not smtp_settings['username'] or not smtp_settings['password']:
        return {"status": "error", "message": "GMAIL_USER and GMAIL_APP_PASSWORD environment variables are required"}
    
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_settings['username']
        msg['To'] = to
        msg['Subject'] = subject
        
        # Add CC if provided
        if cc:
            msg['Cc'] = cc
        
        # Add Reply-To if provided
        if reply_to:
            msg['Reply-To'] = reply_to
            
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_settings['server'], smtp_settings['port']) as server:
            server.starttls()
            server.login(smtp_settings['username'], smtp_settings['password'])
            
            # Include CC recipients in the send_message recipients list if provided
            recipients = [to]
            if cc:
                recipients.append(cc)
                
            server.send_message(msg, to_addrs=recipients)
        
        return {
            "status": "success",
            "message": f"Email sent successfully to {to}" + (f" with CC to {cc}" if cc else ""),
            "to": to,
            "cc": cc,
            "reply_to": reply_to,
            "subject": subject,
            "body": body
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error sending email: {str(e)}",
            "to": to,
            "cc": cc,
            "reply_to": reply_to,
            "subject": subject,
            "body": body
        }

class EmailDrafter:
    def __init__(self, email: str, industry: str, use_gpt: bool = True):
        self.email = email
        self.industry = industry
        self.llm = get_llm(use_gpt)
        self.domain = email.split('@')[1]
        self.company_name = self.domain.split('.')[0]
        
    def research_company(self):
        """Research the company and return findings"""
        # First, try to get direct website data
        website_url = f"www.{self.domain}"
        website_data = extract_website_data(website_url)
        
        # Use the ScrapeWebsiteTool as well for more structured data
        try:
            scraped_data = scrape_tool._run(website_url)
        except Exception as e:
            scraped_data = f"Error using ScrapeWebsiteTool: {str(e)}"
        
        # Combine the data
        direct_website_data = f"""
        DIRECT WEBSITE SCRAPING RESULTS:
        {website_data}
        
        SCRAPE WEBSITE TOOL RESULTS:
        {scraped_data}
        """
        
        # Create research agent with both scraping and search tools
        researcher = Agent(
            role='Company Research Specialist',
            goal='Analyze company and gather comprehensive information',
            backstory=dedent(f"""You are an expert researcher specializing in 
                {self.industry} company analysis. You excel at finding detailed information 
                about companies, their products, and market presence. 
                Use the website data provided first, and supplement with search results."""),
            tools=[search_tool, scrape_tool],
            verbose=True,
            llm=self.llm,
            max_iter=100,
            allow_delegation=False
        )
        
        research_task = Task(
            description=dedent(f"""Research {self.company_name} ({self.domain}) thoroughly.
                Consider their position in the {self.industry} industry.
                
                First, analyze this direct data from their website:
                {direct_website_data}
                
                If the website data is insufficient, use your tools to search for more information.
                
                Step-by-step approach:
                1. Analyze the website data provided
                2. Search for additional company overview and background if needed
                3. Research their products/services in detail
                4. Find information about their team and leadership
                5. Analyze their market position
                6. Identify their tech stack and tools
                
                Focus on:
                - Company's main products/services
                - Value proposition
                - Target market
                - Team information
                - Recent updates or changes
                - Technology stack or tools mentioned
                
                Create a comprehensive profile of the company."""),
            agent=researcher,
            expected_output=dedent("""Detailed company profile including all 
                discovered information in a structured format.""")
        )
        
        # Create and run crew
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True,
            max_rpm=100
        )
        
        result = crew.kickoff()
        return result
    
    def analyze_news(self):
        """Research news and trends about the company"""
        # First, try to extract news from the company website
        website_url = f"www.{self.domain}"
        
        # Try to find common news/blog paths
        news_paths = ['/news', '/blog', '/press', '/media', '/updates', '/articles']
        news_data = ""
        
        for path in news_paths:
            try:
                news_url = f"https://{website_url}{path}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(news_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for article titles and content
                    articles = []
                    
                    # Method 1: Look for article elements
                    for article in soup.find_all(['article', 'div'], class_=lambda c: c and any(x in c for x in ['post', 'article', 'news', 'blog'])):
                        title_elem = article.find(['h1', 'h2', 'h3'])
                        title = title_elem.get_text().strip() if title_elem else "Untitled"
                        
                        date_elem = article.find(['time', 'span', 'div'], class_=lambda c: c and any(x in c for x in ['date', 'time', 'published']))
                        date = date_elem.get_text().strip() if date_elem else ""
                        
                        snippet_elem = article.find(['p', 'div'], class_=lambda c: c and any(x in c for x in ['excerpt', 'summary', 'content']))
                        snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                        
                        if title != "Untitled" or snippet:
                            articles.append(f"Title: {title}\nDate: {date}\nExcerpt: {snippet}\n")
                    
                    # Method 2: If no articles found, look for links that might be news items
                    if not articles:
                        for link in soup.find_all('a', href=True):
                            if link.get_text().strip():
                                articles.append(f"Link: {link.get_text().strip()}\n")
                    
                    # Take only first 10 articles
                    articles = articles[:10]
                    if articles:
                        news_data += f"\nNEWS FROM {news_url}:\n" + "\n".join(articles) + "\n"
            except:
                continue
        
        # Create news analyst agent with both website news and search capability
        news_analyst = Agent(
            role='News and Trends Analyst',
            goal='Find and analyze relevant news and industry trends',
            backstory=dedent(f"""You are skilled at identifying relevant news 
                and understanding {self.industry} industry trends. You can connect company 
                activities to broader market movements. Use website news data if available,
                then supplement with search results."""),
            tools=[search_tool, scrape_tool],
            verbose=True,
            llm=self.llm,
            max_iter=75,
            allow_delegation=False
        )
        
        news_task = Task(
            description=dedent(f"""Research recent news and developments about 
                {self.company_name} and their position in the {self.industry} industry.
                
                First, analyze any news data from their website:
                {news_data}
                
                Then use your tools to search for additional news and trends.
                
                Step-by-step approach:
                1. Analyze any website news data provided
                2. Search for additional company news from the last 3 months
                3. Research {self.industry} industry trends affecting them
                4. Analyze competitor movements
                5. Identify market opportunities
                6. Find any company milestones or achievements
                
                Focus on:
                - Recent company news and press releases
                - Industry trends and developments
                - Competitive landscape
                - Market opportunities and challenges
                - Recent achievements or notable events"""),
            agent=news_analyst,
            expected_output=dedent("""Comprehensive news analysis including 
                company-specific news and relevant industry trends.""")
        )
        
        # Create and run crew
        crew = Crew(
            agents=[news_analyst],
            tasks=[news_task],
            process=Process.sequential,
            verbose=True,
            max_rpm=100
        )
        
        result = crew.kickoff()
        return result
    
    def draft_email(self, research_data: str, news_data: str):
        """Draft email based on research and news"""
        writer = Agent(
            role='Outreach Content Specialist',
            goal='Create highly personalized email content',
            backstory=dedent(f"""You are an expert at crafting personalized 
                outreach emails for {self.industry} companies that resonate with recipients. 
                You excel at combining company research with industry insights.
                You are founder of explainx.ai and your name is Yash Thakker, which 
                is what should be mentioned in the email. Be sure to mention that this email 
                was generated by an AI agent, as a demonstration of your company's AI capabilities."""),
            verbose=True,
            llm=self.llm
        )
        
        # Load template if available
        template = load_email_template(self.industry)
        template_context = f"Use this template as inspiration if available: {template}" if template else ""
        
        email_task = Task(
            description=dedent(f"""Draft a personalized email to {self.email} using 
                the research and news analysis provided. Consider their position in the 
                {self.industry} industry.
                
                The research data is: {research_data}
                
                The news analysis is: {news_data}
                
                Step-by-step approach:
                1. Extract key insights from research
                2. Identify compelling news points
                3. Craft attention-grabbing subject
                4. Write personalized introduction
                5. Present value proposition
                
                Guidelines:
                - Subject line must be a short question with a wave emoji (👋), e.g., "👋 Need AI Agents at {self.company_name}?"
                - Subject should be crisp and direct
                - Keep tone crisp, minimal and professional throughout
                - Be concise - avoid unnecessary words and phrases
                - Get straight to the point quickly
                - Reference specific company details from research
                - Focus on how our AI agents can improve business efficiency for {self.company_name}
                - DO NOT suggest replacing or competing with their existing tools
                - Emphasize complementing and enhancing their current processes
                - Mention relevant {self.industry} trends
                - Focus on value proposition around time/cost savings and business optimization
                - Keep email concise (100-150 words maximum)
                - Include clear call to action
                - Sign as "Jane, AI Agent for Outreach"
                
                IMPORTANT: Your email MUST include the following information:
                1. Start with "I'm Jane, Outreach AI Agent @ ExplainX.ai" before getting into the main content
                2. Clearly state that "This email was drafted by an AI agent" as a demonstration of our technology
                3. Mention that we offer AI agents for business efficiency, including:
                   - Sales automation
                   - Marketing content creation
                   - Customer outreach
                   - Lead qualification
                4. Include 1-2 specific examples of how our AI agents could improve efficiency in sales or marketing for {self.company_name} based on your research
                5. Include a call to action inviting them to reach out if they're interested in implementing AI solutions to enhance their business operations
                6. Do NOT mention any specific individual by name in the email body or signature
                7. Sign simply as "Jane, AI Agent for Outreach"
                
                {template_context}
                
                IMPORTANT: Return your response in valid JSON format with the following structure:
                {{
                    "subject": "Your email subject (5-7 words with emoji)",
                    "body": "Your email body",
                    "cc": "yash@explainx.ai",
                    "reply_to": "yash@explainx.ai"
                }}"""),
            agent=writer,
            expected_output=dedent("""JSON containing subject and body for email.""")
        )
        
        # Create and run crew
        crew = Crew(
            agents=[writer],
            tasks=[email_task],
            process=Process.sequential,
            verbose=True,
            max_rpm=100
        )
        
        result = crew.kickoff()
        
        # Handle CrewOutput object - get the string content from it
        result_str = str(result)
        
        # Try to parse JSON from the result string
        try:
            # Look for JSON in the result string
            import re
            json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                email_content = json.loads(json_str)
                return email_content
            
            # If no JSON pattern found, do simple parsing
            lines = result_str.strip().split('\n')
            subject = ""
            body = ""
            
            for i, line in enumerate(lines):
                if "subject" in line.lower() and ":" in line:
                    subject = line.split(":", 1)[1].strip().strip('"').strip()
                    body_start = i + 1
                    # Look for the body after finding subject
                    for j in range(body_start, len(lines)):
                        if "body" in lines[j].lower() and ":" in lines[j]:
                            body = "\n".join(lines[j+1:]).strip()
                            break
                    if not body:  # If we didn't find a body label, just use everything after subject
                        body = "\n".join(lines[body_start:]).strip()
                    break
            
            # If we found at least a subject, return that
            if subject:
                return {
                    "subject": subject,
                    "body": body
                }
                
            # Fallback to manually creating email content
            return {
                "subject": f"👋 AI efficiency for {self.company_name}?",
                "body": f"I'm Jane, Outreach AI Agent @ ExplainX.ai\n\nI'm reaching out about {self.company_name} in the {self.industry} space.\n\n**This email was drafted by an AI agent.**\n\nWe build AI agents to enhance business efficiency:\n- Sales automation\n- Marketing content creation\n- Customer outreach\n- Lead qualification\n\nOur AI solutions could streamline your sales and marketing operations while complementing your existing tools.\n\nInterested in boosting efficiency? Reply for more details.\n\nJane\nAI Agent for Outreach",
                "cc": "yash@explainx.ai",
                "reply_to": "yash@explainx.ai"
            }
            
        except Exception as e:
            print(f"Error parsing result: {str(e)}")
            # Fallback email content
            return {
                "subject": f"👋 AI efficiency for {self.company_name}?",
                "body": f"I'm Jane, Outreach AI Agent @ ExplainX.ai\n\nI'm reaching out about {self.company_name} in the {self.industry} space.\n\n**This email was drafted by an AI agent.**\n\nWe build AI agents to enhance business efficiency:\n- Sales automation\n- Marketing content creation\n- Customer outreach\n- Lead qualification\n\nOur AI solutions could streamline your sales and marketing operations while complementing your existing tools.\n\nInterested in boosting efficiency? Reply for more details.\n\nJane\nAI Agent for Outreach",
                "cc": "yash@explainx.ai",
                "reply_to": "yash@explainx.ai"
            }

def extract_website_data(url: str) -> str:
    """Extract data from a website using BeautifulSoup"""
    if not url.startswith('http'):
        url = f"https://{url}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else ""
        
        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag and 'content' in meta_tag.attrs:
            meta_desc = meta_tag['content']
        
        # Extract heading text
        headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3']) if h.get_text()]
        heading_text = "\n".join(headings[:10])  # Limit to top 10 headings
        
        # Extract paragraph text
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        paragraph_text = "\n".join(paragraphs[:15])  # Limit to top 15 paragraphs
        
        # Check for about page link
        about_links = soup.find_all('a', href=True, text=lambda t: t and 'about' in t.lower())
        about_urls = [link['href'] for link in about_links]
        
        # Try to extract about page content if found
        about_content = ""
        for about_url in about_urls[:1]:  # Just try the first about link
            if about_url.startswith('/'):
                about_url = url.rstrip('/') + about_url
            elif not about_url.startswith('http'):
                about_url = f"{url.rstrip('/')}/{about_url.lstrip('/')}"
                
            try:
                about_response = requests.get(about_url, headers=headers, timeout=10)
                about_response.raise_for_status()
                about_soup = BeautifulSoup(about_response.text, 'html.parser')
                about_paragraphs = [p.get_text().strip() for p in about_soup.find_all('p') if p.get_text().strip()]
                about_content = "\n".join(about_paragraphs[:10])  # Limit to top 10 paragraphs
            except:
                pass
        
        # Compile all the data
        website_data = f"""
        WEBSITE DATA FOR {url}
        
        TITLE: {title}
        
        META DESCRIPTION: {meta_desc}
        
        MAIN HEADINGS:
        {heading_text}
        
        MAIN CONTENT:
        {paragraph_text}
        
        ABOUT PAGE CONTENT:
        {about_content}
        """
        
        return website_data.strip()
    
    except Exception as e:
        return f"Error extracting data from {url}: {str(e)}"

# Sidebar with settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selection
    model_option = st.radio(
        "Select AI Model",
        ["OpenAI GPT-4", "Local DeepSeek Coder"],
        help="Choose between OpenAI's GPT-4 or local DeepSeek Coder model"
    )
    
    # API Keys
    with st.expander("API Configuration"):
        openai_key = st.text_input("OpenAI API Key", type="password")
        serper_key = st.text_input("Serper API Key", type="password")
        gmail_user = st.text_input("Gmail User", type="password", help="Your Gmail address")
        gmail_password = st.text_input("Gmail App Password", type="password", help="Gmail App Password (NOT your regular password)")
        
        # Save credentials to environment
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if serper_key:
            os.environ["SERPER_API_KEY"] = serper_key
        if gmail_user:
            os.environ["GMAIL_USER"] = gmail_user
        if gmail_password:
            os.environ["GMAIL_APP_PASSWORD"] = gmail_password

# Main content
st.title("📧 Email Preview & Send")
st.caption("Research, draft, preview and send personalized emails")
st.title(f"Email will be sent via {os.getenv('GMAIL_USER')}")

# Input form
with st.form(key="email_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        recipient_email = st.text_input("Recipient Email", help="Enter the prospect's email address")
    
    with col2:
        industry = st.selectbox(
            "Industry",
            ["Technology", "Finance", "Healthcare", "Education", "Other"],
            help="Select the prospect's industry"
        )
    
    submitted = st.form_submit_button("Research & Draft Email")
    
    if submitted:
        if not validate_email(recipient_email):
            st.error("Please enter a valid email address!")
        elif not os.getenv("SERPER_API_KEY"):
            st.error("Please configure Serper API key in the settings!")
        elif model_option == "OpenAI GPT-4" and not os.getenv("OPENAI_API_KEY"):
            st.error("Please configure OpenAI API key in the settings!")
        else:
            # Extract domain for website scraping
            domain = recipient_email.split('@')[1]
            website_url = f"www.{domain}"
            
            # Create email drafter
            drafter = EmailDrafter(
                recipient_email, 
                industry,
                use_gpt=(model_option == "OpenAI GPT-4")
            )
            
            # Check if website is accessible
            try:
                with st.spinner(f"Checking website accessibility: {website_url}"):
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.head(f"https://{website_url}", headers=headers, timeout=5)
                    if response.status_code < 400:
                        st.success(f"✅ Website {website_url} is accessible - will extract data directly")
                        
                        # Extract website data and save to session state
                        website_data = extract_website_data(website_url)
                        st.session_state.website_data = website_data
                        
                        try:
                            scraped_data = scrape_tool._run(website_url)
                            st.session_state.scraped_data = scraped_data
                        except Exception as e:
                            st.session_state.scraped_data = f"Error using ScrapeWebsiteTool: {str(e)}"
                    else:
                        st.warning(f"⚠️ Website {website_url} returned status code {response.status_code} - will rely more on search")
            except Exception as e:
                st.warning(f"⚠️ Could not access {website_url}: {str(e)} - will rely on search")
            
            # Research phase
            with st.spinner("Researching company..."):
                research_data = drafter.research_company()
                st.session_state.research_data = research_data
            
            # News analysis phase
            with st.spinner("Analyzing news and trends..."):
                news_data = drafter.analyze_news()
                st.session_state.news_data = news_data
            
            # Draft email
            with st.spinner("Drafting personalized email..."):
                email_content = drafter.draft_email(research_data, news_data)
                st.session_state.draft_subject = email_content.get("subject", "")
                st.session_state.draft_body = email_content.get("body", "")
                st.session_state.draft_cc = email_content.get("cc", "yash@explainx.ai")
                st.session_state.draft_reply_to = email_content.get("reply_to", "yash@explainx.ai")
                st.session_state.email_data = {
                    "to": recipient_email,
                    "industry": industry
                }
            
            st.success("Email drafted successfully! Preview it below.")

# Display preview if email is drafted
if st.session_state.email_data and st.session_state.draft_subject and st.session_state.draft_body:
    st.header("Email Preview")
    
    # Research and News tabs for reference
    tab1, tab2, tab3, tab4 = st.tabs(["Email Draft", "Company Research", "News Analysis", "Website Data"])
    
    with tab1:
        # Editable email content
        to_email = st.session_state.email_data["to"]
        edited_subject = st.text_input("Subject", value=st.session_state.draft_subject)
        edited_body = st.text_area("Body", value=st.session_state.draft_body, height=300)
        
        # Add fields for CC and Reply-To
        cc_email = st.text_input("CC", value=st.session_state.draft_cc)
        reply_to_email = st.text_input("Reply-To", value=st.session_state.draft_reply_to)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Send Email"):
                if not os.getenv("GMAIL_USER") or not os.getenv("GMAIL_APP_PASSWORD"):
                    st.error("Please configure Gmail credentials in the settings!")
                else:
                    with st.spinner("Sending email..."):
                        result = send_email(
                            st.session_state.email_data["to"],
                            edited_subject,
                            edited_body,
                            cc=cc_email,
                            reply_to=reply_to_email
                        )
                        
                        if result["status"] == "success":
                            st.success(result["message"])
                            st.session_state.email_sent = True
                        else:
                            st.error(result["message"])
        
        with col2:
            if st.button("Regenerate Email"):
                if st.session_state.research_data and st.session_state.news_data:
                    with st.spinner("Regenerating email..."):
                        # Create new drafter
                        drafter = EmailDrafter(
                            st.session_state.email_data["to"],
                            st.session_state.email_data["industry"],
                            use_gpt=(model_option == "OpenAI GPT-4")
                        )
                        
                        # Draft new email with existing research data
                        email_content = drafter.draft_email(
                            st.session_state.research_data, 
                            st.session_state.news_data
                        )
                        
                        # Update session state
                        st.session_state.draft_subject = email_content.get("subject", "")
                        st.session_state.draft_body = email_content.get("body", "")
                        st.session_state.draft_cc = email_content.get("cc", "yash@explainx.ai")
                        st.session_state.draft_reply_to = email_content.get("reply_to", "yash@explainx.ai")
                        
                        # Rerun to update UI
                        st.rerun()
        
        with col3:
            if st.button("Reset"):
                st.session_state.email_data = None
                st.session_state.research_data = None
                st.session_state.news_data = None
                st.session_state.website_data = None
                st.session_state.scraped_data = None
                st.session_state.draft_subject = ""
                st.session_state.draft_body = ""
                st.session_state.email_sent = False
                st.rerun()
    
    with tab2:
        if st.session_state.research_data:
            st.markdown("### Company Research")
            st.markdown(st.session_state.research_data)
    
    with tab3:
        if st.session_state.news_data:
            st.markdown("### News Analysis")
            st.markdown(st.session_state.news_data)
    
    with tab4:
        st.markdown("### Website Data")
        domain = st.session_state.email_data["to"].split('@')[1]
        website_url = f"www.{domain}"
        
        if st.button("Refresh Website Data"):
            with st.spinner(f"Extracting data from {website_url}..."):
                website_data = extract_website_data(website_url)
                st.session_state.website_data = website_data
                
                try:
                    scraped_data = scrape_tool._run(website_url)
                    st.session_state.scraped_data = scraped_data
                except Exception as e:
                    st.session_state.scraped_data = f"Error using ScrapeWebsiteTool: {str(e)}"
        
        # Direct BS4 extraction
        if 'website_data' in st.session_state:
            with st.expander("BS4 Extraction Results", expanded=True):
                st.text(st.session_state.website_data)
        
        # ScrapeWebsiteTool results
        if 'scraped_data' in st.session_state:
            with st.expander("ScrapeWebsiteTool Results", expanded=True):
                st.text(st.session_state.scraped_data)
            
else:
    st.info("Enter a valid email address and click 'Research & Draft Email' to get started.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by @goyashy")

# Help section
with st.sidebar:
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### How to use this tool
        1. Configure your API keys in Settings
        2. Enter a prospect's email address
        3. Select their industry
        4. Click 'Research & Draft Email'
        5. Review the research and draft
        6. Edit the email if needed
        7. Click 'Send Email' when ready
        
        ### Requirements
        - Gmail account with App Password
        - Serper API key for research
        - OpenAI API key (if using GPT-4)
        
        ### Need Help?
        Contact @goyashy for support
        """) 
        