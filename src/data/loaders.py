"""
Data loaders for CV RAG System.
Supports loading CVs from CSV and Parquet files.
"""

import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


def load_cvs(
    file_path: str = "./data/cvs.csv",
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load CVs from a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file containing CVs.
        required_columns: List of required column names. 
                         Defaults to ["candidate_id", "raw_text"].
    
    Returns:
        DataFrame with CV data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    if required_columns is None:
        required_columns = ["candidate_id", "raw_text"]
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")
    
    logger.info(f"Loading CVs from {file_path}")
    
    # Load CSV
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Validate required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean up
    df = df.dropna(subset=["candidate_id", "raw_text"])
    df["candidate_id"] = df["candidate_id"].astype(str)
    
    logger.info(f"Loaded {len(df)} CVs")
    
    return df


def load_cvs_from_parquet(
    file_path: str = "./data/cvs.parquet",
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load CVs from a Parquet file into a pandas DataFrame.
    
    Args:
        file_path: Path to the Parquet file containing CVs.
        required_columns: List of required column names.
                         Defaults to ["candidate_id", "raw_text"].
    
    Returns:
        DataFrame with CV data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    if required_columns is None:
        required_columns = ["candidate_id", "raw_text"]
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")
    
    logger.info(f"Loading CVs from {file_path}")
    
    # Load Parquet
    df = pd.read_parquet(file_path)
    
    # Validate required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean up
    df = df.dropna(subset=["candidate_id", "raw_text"])
    df["candidate_id"] = df["candidate_id"].astype(str)
    
    logger.info(f"Loaded {len(df)} CVs")
    
    return df


def load_cvs_auto(file_path: str) -> pd.DataFrame:
    """
    Automatically load CVs based on file extension.
    
    Args:
        file_path: Path to the CV file (CSV or Parquet).
    
    Returns:
        DataFrame with CV data.
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == ".parquet":
        return load_cvs_from_parquet(str(file_path))
    else:
        return load_cvs(str(file_path))


def create_sample_cv_data() -> pd.DataFrame:
    """
    Create sample CV data for testing/demo purposes.
    
    Returns:
        DataFrame with sample CV data.
    """
    sample_cvs = [
        {
            "candidate_id": "CV001",
            "name": "Alice Johnson",
            "email": "alice.johnson@email.com",
            "role": "Senior Software Engineer",
            "location": "San Francisco, CA",
            "years_experience": 8,
            "raw_text": """
ALICE JOHNSON
Senior Software Engineer | San Francisco, CA
alice.johnson@email.com | (555) 123-4567 | linkedin.com/in/alicejohnson

SUMMARY
Experienced software engineer with 8+ years of expertise in building scalable web applications and distributed systems. Strong background in Python, JavaScript, and cloud technologies. Passionate about clean code and mentoring junior developers.

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2020 - Present
- Led development of microservices architecture serving 10M+ daily users
- Implemented CI/CD pipelines reducing deployment time by 60%
- Mentored team of 5 junior developers
- Technologies: Python, FastAPI, Kubernetes, AWS, PostgreSQL

Software Engineer | StartupXYZ | 2017 - 2020
- Built real-time data processing pipeline handling 1M events/hour
- Developed RESTful APIs for mobile and web clients
- Reduced infrastructure costs by 40% through optimization
- Technologies: Python, Django, Redis, Docker, GCP

Junior Developer | WebAgency | 2015 - 2017
- Developed responsive web applications for various clients
- Created automated testing suites improving code quality
- Technologies: JavaScript, React, Node.js, MongoDB

EDUCATION
Master of Science in Computer Science | Stanford University | 2015
Bachelor of Science in Computer Science | UC Berkeley | 2013

SKILLS
Programming: Python, JavaScript, TypeScript, Go, SQL
Frameworks: FastAPI, Django, React, Node.js, Express
Cloud & DevOps: AWS, GCP, Kubernetes, Docker, Terraform, CI/CD
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch

CERTIFICATIONS
- AWS Solutions Architect Professional
- Google Cloud Professional Data Engineer
- Kubernetes Administrator (CKA)

PROJECTS
Open Source Contributor - Contributed to multiple Python libraries including FastAPI and Pydantic
ML Pipeline Framework - Built internal framework for deploying ML models at scale
""",
        },
        {
            "candidate_id": "CV002",
            "name": "Bob Martinez",
            "email": "bob.martinez@email.com",
            "role": "Data Scientist",
            "location": "New York, NY",
            "years_experience": 5,
            "raw_text": """
BOB MARTINEZ
Data Scientist | New York, NY
bob.martinez@email.com | (555) 987-6543 | github.com/bobmartinez

PROFILE
Data Scientist with 5 years of experience in machine learning, statistical modeling, and data analysis. Expertise in NLP, computer vision, and recommendation systems. Strong communicator who bridges the gap between technical teams and business stakeholders.

EXPERIENCE

Senior Data Scientist | DataDriven Corp | 2021 - Present
- Developed NLP models for sentiment analysis achieving 94% accuracy
- Built recommendation engine increasing user engagement by 35%
- Led A/B testing framework used across 20+ product experiments
- Technologies: Python, PyTorch, TensorFlow, Spark, Airflow

Data Scientist | AnalyticsPro | 2019 - 2021
- Created customer churn prediction model saving $2M annually
- Implemented computer vision pipeline for quality control
- Built automated reporting dashboards for executive team
- Technologies: Python, Scikit-learn, Pandas, Tableau, SQL

Data Analyst | FinanceHub | 2018 - 2019
- Performed statistical analysis on financial datasets
- Created data visualization for quarterly reports
- Automated data collection processes
- Technologies: Python, R, Excel, Power BI

EDUCATION
PhD in Statistics (ABD) | Columbia University | 2018
Master of Science in Applied Mathematics | NYU | 2016
Bachelor of Science in Mathematics | MIT | 2014

SKILLS
ML/AI: Deep Learning, NLP, Computer Vision, Recommendation Systems
Tools: Python, R, PyTorch, TensorFlow, Scikit-learn, Spark
Data: SQL, Pandas, NumPy, Matplotlib, Tableau, Power BI
Cloud: AWS SageMaker, GCP AI Platform, Databricks

PUBLICATIONS
- "Transformer-based Approach for Domain-Specific NER" - ACL 2022
- "Scalable Recommendation Systems Using Graph Neural Networks" - RecSys 2021

AWARDS
- Best Paper Award - Data Science Conference 2021
- Innovation Award - DataDriven Corp 2022
""",
        },
        {
            "candidate_id": "CV003",
            "name": "Carol Chen",
            "email": "carol.chen@email.com",
            "role": "Product Manager",
            "location": "Seattle, WA",
            "years_experience": 6,
            "raw_text": """
CAROL CHEN
Product Manager | Seattle, WA
carol.chen@email.com | (555) 456-7890 | linkedin.com/in/carolchen

SUMMARY
Strategic product manager with 6 years of experience driving product development from conception to launch. Track record of delivering products that delight users and drive business growth. Strong technical background combined with business acumen.

EXPERIENCE

Senior Product Manager | CloudTech Solutions | 2021 - Present
- Launched enterprise SaaS product generating $5M ARR in first year
- Managed cross-functional team of 15 engineers, designers, and analysts
- Conducted 100+ customer interviews to validate product-market fit
- Increased user retention by 45% through data-driven feature prioritization

Product Manager | InnovateCo | 2019 - 2021
- Led mobile app redesign resulting in 4.8 star rating (up from 3.2)
- Defined and executed product roadmap for 3 product lines
- Implemented agile methodologies improving sprint velocity by 30%
- Collaborated with engineering to reduce technical debt by 25%

Associate Product Manager | TechStart | 2017 - 2019
- Assisted in launching 2 successful products from MVP to scale
- Created user stories and acceptance criteria for development team
- Analyzed user behavior data to inform product decisions
- Coordinated beta testing programs with 500+ users

EDUCATION
MBA | University of Washington | 2017
Bachelor of Science in Industrial Engineering | Georgia Tech | 2015

SKILLS
Product: Roadmapping, User Research, A/B Testing, Analytics, Agile/Scrum
Technical: SQL, Python (basic), Jira, Amplitude, Mixpanel
Business: P&L Management, Go-to-Market Strategy, Stakeholder Management

CERTIFICATIONS
- Certified Scrum Product Owner (CSPO)
- Google Analytics Certified
- Product Management Certificate - Product School

PROJECTS
Customer Feedback Platform - Built internal tool for aggregating customer feedback
Product Analytics Dashboard - Created executive dashboard for product metrics
""",
        },
        {
            "candidate_id": "CV004",
            "name": "David Kim",
            "email": "david.kim@email.com",
            "role": "DevOps Engineer",
            "location": "Austin, TX",
            "years_experience": 7,
            "raw_text": """
DAVID KIM
DevOps Engineer | Austin, TX
david.kim@email.com | (555) 321-0987 | github.com/davidkim

PROFILE
DevOps engineer with 7 years of experience in cloud infrastructure, automation, and site reliability. Expert in building and maintaining large-scale distributed systems. Passionate about infrastructure as code and continuous improvement.

EXPERIENCE

Staff DevOps Engineer | ScaleUp Systems | 2021 - Present
- Architected multi-region Kubernetes infrastructure handling 50K RPS
- Reduced infrastructure costs by 35% through resource optimization
- Implemented zero-downtime deployment strategies
- Led incident response and established SRE practices
- Technologies: Kubernetes, Terraform, AWS, Prometheus, Grafana

Senior DevOps Engineer | CloudNative Inc | 2018 - 2021
- Built CI/CD pipelines for 50+ microservices
- Migrated legacy infrastructure to containerized environment
- Implemented infrastructure as code reducing provisioning time by 80%
- Technologies: Docker, Jenkins, Ansible, GCP, ELK Stack

DevOps Engineer | WebScale Corp | 2016 - 2018
- Managed AWS infrastructure for e-commerce platform
- Implemented monitoring and alerting systems
- Automated backup and disaster recovery procedures
- Technologies: AWS, CloudFormation, Python, Bash

EDUCATION
Bachelor of Science in Computer Engineering | UT Austin | 2016

SKILLS
Cloud: AWS (Expert), GCP, Azure
Containers: Kubernetes, Docker, Helm, Istio
IaC: Terraform, Pulumi, CloudFormation, Ansible
CI/CD: Jenkins, GitLab CI, GitHub Actions, ArgoCD
Monitoring: Prometheus, Grafana, Datadog, PagerDuty
Languages: Python, Go, Bash, YAML

CERTIFICATIONS
- AWS Solutions Architect Professional
- Certified Kubernetes Administrator (CKA)
- Certified Kubernetes Security Specialist (CKS)
- HashiCorp Terraform Associate

PROJECTS
K8s Cost Optimizer - Open source tool for Kubernetes cost optimization
GitOps Framework - Internal framework for standardized GitOps deployments
""",
        },
        {
            "candidate_id": "CV005",
            "name": "Emma Wilson",
            "email": "emma.wilson@email.com",
            "role": "UX Designer",
            "location": "Los Angeles, CA",
            "years_experience": 4,
            "raw_text": """
EMMA WILSON
UX Designer | Los Angeles, CA
emma.wilson@email.com | (555) 654-3210 | portfolio.emmawilson.design

SUMMARY
Creative UX designer with 4 years of experience crafting intuitive digital experiences. Skilled in user research, interaction design, and design systems. Advocate for accessibility and inclusive design practices.

EXPERIENCE

Senior UX Designer | DesignForward Agency | 2022 - Present
- Led UX for enterprise software redesign serving 100K+ users
- Established design system reducing design-to-development time by 40%
- Conducted usability studies with 200+ participants
- Mentored 2 junior designers

UX Designer | AppCreate Studio | 2020 - 2022
- Designed mobile apps with combined 1M+ downloads
- Created wireframes, prototypes, and high-fidelity mockups
- Collaborated with product and engineering teams in agile environment
- Improved task completion rate by 25% through iterative testing

Junior UX Designer | Digital Agency XY | 2019 - 2020
- Supported senior designers on client projects
- Conducted competitive analysis and user research
- Created user personas and journey maps

EDUCATION
Bachelor of Fine Arts in Graphic Design | ArtCenter College of Design | 2019

SKILLS
Design: Figma, Sketch, Adobe Creative Suite, Principle, Framer
Research: User Interviews, Usability Testing, A/B Testing, Analytics
Methods: Design Thinking, Jobs-to-be-Done, Accessibility (WCAG)
Technical: HTML/CSS basics, Design Systems, Component Libraries

CERTIFICATIONS
- Google UX Design Certificate
- Certified Usability Analyst (CUA)
- Accessibility Specialist Certification

AWARDS
- Webby Award Honoree 2023 - Mobile App Design
- AIGA Design Award 2022

PROJECTS
Inclusive Design Toolkit - Created open-source accessibility checklist
Mobile Banking Redesign - Case study featured in UX Collective
""",
        },
    ]
    
    return pd.DataFrame(sample_cvs)


def save_sample_data(output_path: str = "./data/cvs.csv") -> None:
    """
    Generate and save sample CV data to a file.
    
    Args:
        output_path: Path where to save the sample data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = create_sample_cv_data()
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Saved sample CV data to {output_path}")


def df_from_pdf(
    pdf_bytes: bytes,
    candidate_id: str,
    filename: Optional[str] = None,
    **metadata,
) -> pd.DataFrame:
    """
    Convert PDF bytes to a DataFrame row matching the CV pipeline schema.
    
    This function extracts text from a PDF and creates a single-row DataFrame
    with the same schema expected by the chunking pipeline.
    
    Args:
        pdf_bytes: Raw bytes of the PDF file.
        candidate_id: Unique identifier for the candidate.
        filename: Original filename (optional, stored in metadata).
        **metadata: Additional metadata fields (e.g., name, email, role).
    
    Returns:
        DataFrame with a single row containing:
        - candidate_id: The provided candidate ID
        - raw_text: Extracted text from PDF
        - filename: Original filename (if provided)
        - Any additional metadata fields
    
    Raises:
        ValueError: If text extraction yields empty result.
        ImportError: If pypdf is not installed.
    """
    from src.data.pdf_utils import extract_text_from_pdf_bytes
    
    # Extract text from PDF
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    
    if not raw_text or len(raw_text.strip()) < 50:
        raise ValueError(
            "PDF text extraction yielded insufficient text. "
            "The PDF might be scanned/image-based and requires OCR."
        )
    
    # Build the row data
    row_data = {
        "candidate_id": candidate_id,
        "raw_text": raw_text,
    }
    
    # Add filename if provided
    if filename:
        row_data["filename"] = filename
    
    # Add any additional metadata
    row_data.update(metadata)
    
    logger.info(
        f"Created DataFrame row from PDF: candidate_id={candidate_id}, "
        f"text_length={len(raw_text)}"
    )
    
    return pd.DataFrame([row_data])

