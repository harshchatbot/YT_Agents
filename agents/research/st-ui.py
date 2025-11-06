import streamlit as st
import pandas as pd
from typing import List
from with_st import DetailedSalesCrew  # Import from your main implementation file
import json
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Sales Outreach Platform",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'emails' not in st.session_state:
    st.session_state.emails = []
if 'results' not in st.session_state:
    st.session_state.results = []

def validate_email(email: str) -> bool:
    """Simple email validation"""
    return '@' in email and '.' in email.split('@')[1]

def save_email_template(industry: str, template: str):
    """Save email template to templates directory"""
    os.makedirs('email_templates', exist_ok=True)
    with open(f'email_templates/{industry.lower()}.txt', 'w') as f:
        f.write(template)

def load_email_template(industry: str) -> str:
    """Load email template for given industry"""
    try:
        with open(f'email_templates/{industry.lower()}.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def run_sales_crew(emails: List[dict], use_gpt: bool = False) -> List[dict]:
    """Run the sales crew with the given emails"""
    sales_crew = DetailedSalesCrew(emails, use_gpt)
    return sales_crew.run()

# Sidebar
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
st.title("🤖 AI Sales Outreach Platform")
st.caption("Personalized email outreach powered by AI")

# Tab selection
tab1, tab2, tab3 = st.tabs(["Add Prospects", "Email Templates", "Results"])

# Add Prospects Tab
with tab1:
    st.header("Add Target Prospects")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email = st.text_input("Email Address", help="Enter the prospect's email address")
        industry = st.selectbox(
            "Industry",
            ["Technology", "Finance", "Healthcare", "Education", "Other"],
            help="Select the prospect's industry"
        )
        
        upload_file = st.file_uploader("Or Upload CSV", type=['csv'], 
            help="CSV should have 'email' and 'industry' columns")
        
        if upload_file is not None:
            try:
                df = pd.read_csv(upload_file)
                if 'email' in df.columns and 'industry' in df.columns:
                    new_prospects = df[['email', 'industry']].to_dict('records')
                    for prospect in new_prospects:
                        if validate_email(prospect['email']) and \
                           not any(e["email"] == prospect['email'] for e in st.session_state.emails):
                            st.session_state.emails.append(prospect)
                    st.success(f"Added {len(new_prospects)} prospects from CSV!")
                else:
                    st.error("CSV must contain 'email' and 'industry' columns")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    with col2:
        if st.button("Add Prospect", help="Add single prospect to the list"):
            if validate_email(email):
                if not any(e["email"] == email for e in st.session_state.emails):
                    st.session_state.emails.append({"email": email, "industry": industry})
                    st.success(f"Added {email} to the prospect list!")
                else:
                    st.warning("This email is already in the list!")
            else:
                st.error("Please enter a valid email address!")
    
    # Display current prospects
    if st.session_state.emails:
        st.subheader("Current Prospects")
        df = pd.DataFrame(st.session_state.emails)
        st.dataframe(df, hide_index=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Clear List", help="Remove all prospects from the list"):
                st.session_state.emails = []
                st.success("Prospect list cleared!")
        
        with col4:
            if st.button("Run Outreach Campaign", help="Start sending personalized emails"):
                if not os.getenv("GMAIL_USER") or not os.getenv("GMAIL_APP_PASSWORD"):
                    st.error("Please configure Gmail credentials in the settings!")
                elif not os.getenv("SERPER_API_KEY"):
                    st.error("Please configure Serper API key in the settings!")
                elif model_option == "OpenAI GPT-4" and not os.getenv("OPENAI_API_KEY"):
                    st.error("Please configure OpenAI API key in the settings!")
                else:
                    with st.spinner("Running outreach campaign..."):
                        try:
                            results = run_sales_crew(
                                st.session_state.emails,
                                use_gpt=(model_option == "OpenAI GPT-4")
                            )
                            st.session_state.results = results
                            st.success("Campaign completed successfully!")
                        except Exception as e:
                            st.error(f"Error running campaign: {str(e)}")

# Email Templates Tab
with tab2:
    st.header("Email Templates")
    
    template_industry = st.selectbox(
        "Select Industry for Template",
        ["Technology", "Finance", "Healthcare", "Education", "Other"],
        key="template_industry"
    )
    
    existing_template = load_email_template(template_industry)
    
    template_content = st.text_area(
        "Email Template",
        value=existing_template,
        height=300,
        help="Use placeholders: {company}, {industry}, {name}, etc."
    )
    
    with st.expander("Template Variables Help"):
        st.markdown("""
        Available template variables:
        - `{company}`: Company name
        - `{industry}`: Industry name
        - `{domain}`: Company domain
        - `{name}`: Recipient's name (if available)
        
        Example template:
        ```
        Subject: Enhancing {company}'s Capabilities
        
        Hi {name},
        
        I noticed {company}'s impressive work in the {industry} sector...
        
        Best regards,
        Yash Thakker
        Founder, ExplainX.ai
        ```
        """)
    
    if st.button("Save Template"):
        save_email_template(template_industry, template_content)
        st.success(f"Template saved for {template_industry} industry!")

# Results Tab
with tab3:
    st.header("Campaign Results")
    
    if st.session_state.results:
        # Download results button
        results_df = pd.DataFrame([{
            'email': r['email'],
            'industry': r['industry'],
            'status': 'success' if json.loads(r['result'])['status'] == 'success' else 'error',
            'subject': json.loads(r['result'])['subject'],
            'body': json.loads(r['result'])['body']
        } for r in st.session_state.results])
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            csv,
            "campaign_results.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Display individual results
        for result in st.session_state.results:
            with st.expander(f"Results for {result['email']} ({result['industry']})"):
                try:
                    # Parse the JSON result
                    email_content = json.loads(result['result'])
                    
                    # Show status
                    if "status" in email_content:
                        st.success(email_content["message"])
                    elif "error" in email_content:
                        st.error(email_content["error"])
                    
                    # Show email content
                    st.subheader("Subject")
                    st.write(email_content.get('subject', 'N/A'))
                    st.subheader("Body")
                    st.write(email_content.get('body', 'N/A'))
                    
                except Exception as e:
                    st.error(f"Error displaying result: {str(e)}")
                    st.write(result['result'])
    else:
        st.info("No campaign results yet. Run an outreach campaign to see results here.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by @goyashy")

# Add help/documentation tooltip
with st.sidebar:
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### Quick Start Guide
        1. Configure your API keys in Settings
        2. Add prospects individually or via CSV
        3. Optionally set up email templates
        4. Run the campaign
        
        ### Requirements
        - Gmail account with App Password
        - Serper API key for research
        - OpenAI API key (if using GPT-4)
        
        ### Need Help?
        Contact @goyashy for support
        """)