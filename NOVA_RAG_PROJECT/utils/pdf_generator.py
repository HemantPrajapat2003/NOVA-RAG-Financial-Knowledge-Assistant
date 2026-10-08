"""
Utility script to generate realistic, professional financial PDF documents
for the AI Financial Knowledge Assistant using ReportLab.
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y'."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "APEX TRUST BANK | CONFIDENTIAL FINANCIAL POLICY MANUAL")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "Apex Financial Knowledge Base — Internal & Customer Advisory")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        
        self.restoreState()


def get_styles():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=4,
        spaceAfter=6
    )
    
    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'h1': h1_style,
        'h2': h2_style,
        'body': body_style,
        'bullet': bullet_style,
        'callout': callout_style
    }


def build_pdf(filepath, title, subtitle, pages_content):
    """
    Builds a multi-page PDF document.
    pages_content is a list of lists of flowables for each page.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = get_styles()
    story = []
    
    # Document Title Block
    story.append(Paragraph(title, styles['title']))
    story.append(Paragraph(subtitle, styles['subtitle']))
    story.append(Spacer(1, 10))
    
    for i, page_items in enumerate(pages_content):
        if i > 0:
            story.append(PageBreak())
        for item in page_items:
            story.append(item)
            
    doc.build(story, canvasmaker=NumberedCanvas)


def generate_all_sample_pdfs(target_dir="documents"):
    """
    Generates all 8 standard financial PDF documents with authentic financial clauses,
    explicit tables, and structured pages.
    """
    os.makedirs(target_dir, exist_ok=True)
    styles = get_styles()
    
    # ==========================================
    # 1. Home_Loan_Policy.pdf
    # ==========================================
    hl_p1 = [
        Paragraph("1. Purpose and Operational Scope", styles['h1']),
        Paragraph("This policy manual defines underwriting principles, eligibility benchmarks, and documentation standards for retail residential home loans sanctioned by Apex Trust Bank. All credit appraisal officers and branch operations managers must adhere strictly to these guidelines.", styles['body']),
        
        Paragraph("2. Applicant Eligibility Criteria & Age Requirements", styles['h1']),
        Paragraph("Credit assessment of home loan applicants relies on demographic stability, debt-service capacity, and validated credit history.", styles['body']),
        Paragraph("<b>• Minimum Age for Home Loan:</b> The primary applicant must be at least <b>21 years of age</b> at the date of loan application.", styles['bullet']),
        Paragraph("<b>• Maximum Age at Maturity:</b> For salaried applicants, the maximum permissible age is <b>65 years</b> or official retirement age (whichever is earlier). For self-employed professionals and non-professionals, the maximum age limit is <b>70 years</b> at loan maturity.", styles['bullet']),
        Paragraph("<b>• Minimum Net Monthly Income:</b> ₹35,000 net monthly take-home income for salaried individuals. For self-employed applicants, a minimum audited net profit after tax of ₹5,00,000 per annum is mandatory.", styles['bullet']),
        Paragraph("<b>• Credit Score (CIBIL):</b> Minimum credit score of 750 for base lending rate eligibility. Scores between 700 and 749 are acceptable with a risk spread premium of 0.25%. Applications with CIBIL score below 700 require Zonal Underwriting Committee approval.", styles['bullet']),
        
        Spacer(1, 8),
        Paragraph("3. Loan-to-Value (LTV) Ratios and Sanction Caps", styles['h1']),
        Paragraph("Loan amounts are constrained by standard prudential LTV ratios linked to ticket size:", styles['body']),
        Paragraph("<b>• Up to ₹30 Lakhs:</b> Maximum LTV ratio of <b>90%</b> of the agreement value.", styles['bullet']),
        Paragraph("<b>• ₹30 Lakhs to ₹75 Lakhs:</b> Maximum LTV ratio of <b>80%</b> of the agreement value.", styles['bullet']),
        Paragraph("<b>• Above ₹75 Lakhs:</b> Maximum LTV ratio of <b>75%</b> of the agreement value.", styles['bullet']),
    ]
    
    hl_p2 = [
        Paragraph("4. Mandatory Documentation Checklist for Home Loan", styles['h1']),
        Paragraph("Applicants must submit verified copies of all requisite documents before sanction and legal disbursement:", styles['body']),
        
        Paragraph("A. Know Your Customer (KYC) Identification Proofs:", styles['h2']),
        Paragraph("• Valid Photo Identity Proof: Permanent Account Number (PAN) Card (mandatory for financial vetting), along with Passport, Voter ID Card, or Aadhaar Card with biometric validation.", styles['bullet']),
        Paragraph("• Proof of Current Residential Address: Registered Rent Agreement, Electricity Bill, Water Bill (not older than 2 months), or Aadhaar Card.", styles['bullet']),
        
        Paragraph("B. Financial & Income Documents (Salaried Borrowers):", styles['h2']),
        Paragraph("• Last 3 months' certified salary pay slips mentioning all statutory deductions.", styles['bullet']),
        Paragraph("• Form 16 (Part A and Part B) for the preceding 2 financial years.", styles['bullet']),
        Paragraph("• Last 6 months' bank operative salary account statements evidencing direct salary credits.", styles['bullet']),
        Paragraph("• Employment continuity verification certificate or Appointment Letter.", styles['bullet']),
        
        Paragraph("C. Financial Documents (Self-Employed / Business Owners):", styles['h2']),
        Paragraph("• Income Tax Returns (ITR) alongside computation of total income for the last 3 financial years.", styles['bullet']),
        Paragraph("• Audited Balance Sheet and Profit & Loss accounts with Tax Audit Report signed by a Chartered Accountant.", styles['bullet']),
        Paragraph("• Last 12 months' operative bank account statements for current and savings accounts.", styles['bullet']),
        Paragraph("• Business proof: GST Registration Certificate, Shop & Establishment License, or Certificate of Incorporation.", styles['bullet']),
        
        Paragraph("D. Property and Title Verification Documents:", styles['h2']),
        Paragraph("• Registered Sale Agreement / Builder Buyer Agreement stamped as per state stamp act.", styles['bullet']),
        Paragraph("• Allotment Letter from Developer / Housing Board or Society.", styles['bullet']),
        Paragraph("• Clear title search report and Encumbrance Certificate (EC) for the past 30 years.", styles['bullet']),
        Paragraph("• Sanctioned building architectural blueprint approved by competent municipal authority.", styles['bullet']),
        Paragraph("• No Objection Certificate (NOC) from housing society or builder.", styles['bullet']),
        
        Spacer(1, 8),
        Paragraph("5. Foreclosure Norms & Prepayment Charges", styles['h1']),
        Paragraph("In accordance with regulatory directives, Apex Trust Bank imposes <b>zero (0%) foreclosure penalty and prepayment charges</b> on all floating interest rate home loans sanctioned to individual borrowers, irrespective of whether funded via own sources or refinancing.", styles['body']),
    ]
    
    build_pdf(
        os.path.join(target_dir, "Home_Loan_Policy.pdf"),
        "APEX TRUST BANK - RETAIL HOME LOAN POLICY",
        "Document Type: Loan Policy | Reference: ATB/POL/RET-HL/2026-V4",
        [hl_p1, hl_p2]
    )

    # ==========================================
    # 2. Personal_Loan_Policy.pdf
    # ==========================================
    pl_p1 = [
        Paragraph("1. Personal Loan Scheme Overview", styles['h1']),
        Paragraph("Apex Trust Bank offers unsecured multi-purpose personal credit facilities designed for salaried and self-employed professionals to meet emergent capital, medical, travel, or debt-consolidation requirements.", styles['body']),
        
        Paragraph("2. Eligibility Norms", styles['h1']),
        Paragraph("<b>• Age Limit:</b> Minimum age of <b>23 years</b> at loan sanction; maximum age of <b>58 years</b> (for salaried) or <b>65 years</b> (for self-employed) at scheduled loan maturity.", styles['bullet']),
        Paragraph("<b>• Employment & Income Stability:</b> Minimum 2 continuous years of overall employment with at least 1 full year with current organization. Minimum monthly take-home salary of ₹30,000.", styles['bullet']),
        Paragraph("<b>• Maximum Fixed Obligation to Income Ratio (FOIR):</b> Aggregate existing and proposed EMIs must not exceed 50% of verified net monthly earnings.", styles['bullet']),
        Paragraph("<b>• Loan Quantum:</b> Sanction limits range from a minimum of ₹50,000 to a maximum of ₹25,00,000, with repayment tenures between 12 and 60 months.", styles['bullet']),
        
        Spacer(1, 10),
        Paragraph("3. Schedule of Foreclosure Charges and Prepayment Rules", styles['h1']),
        Paragraph("Unsecured personal loans are governed by defined lock-in covenants and structured foreclosure tariffs:", styles['body']),
        Paragraph("<b>• Mandatory Lock-in Period:</b> Foreclosure or partial prepayment is strictly prohibited during the initial <b>6 months</b> from the date of initial disbursement. Any prepayment request prior to completion of 6 installments will be rejected.", styles['bullet']),
        Paragraph("<b>• Foreclosure Charges (Between 6 to 12 Months):</b> If the borrower settles the loan after 6 months but within 12 months of disbursement, a foreclosure charge of <b>4.0% of the outstanding principal amount</b> plus applicable GST is levied.", styles['bullet']),
        Paragraph("<b>• Foreclosure Charges (Between 13 to 24 Months):</b> Foreclosure completed between month 13 and month 24 incurs a foreclosure charge of <b>3.0% of the outstanding principal amount</b> plus GST.", styles['bullet']),
        Paragraph("<b>• Foreclosure Charges (Between 25 to 36 Months):</b> Foreclosure completed between month 25 and month 36 incurs a foreclosure charge of <b>2.0% of the outstanding principal amount</b> plus GST.", styles['bullet']),
        Paragraph("<b>• Foreclosure Charges Beyond 36 Months:</b> For loans prepaid after 36 months, the foreclosure charge is reduced to <b>1.0% of the outstanding principal amount</b> plus GST.", styles['bullet']),
        Paragraph("<b>• Part-Prepayment Terms:</b> Allowed twice per financial calendar year after the 6-month lock-in. Minimum part payment amount is 2 EMIs, maximum 25% of principal balance per annum, subject to a 2% part-payment fee.", styles['bullet']),
    ]
    
    pl_p2 = [
        Paragraph("4. Interest Rate Structure and Processing Fees", styles['h1']),
        Paragraph("• Annual Percentage Rate (APR) spreads between 10.75% and 18.50% p.a., determined dynamically based on applicant risk rating, credit bureau record, and corporate categorization.", styles['body']),
        Paragraph("• One-time non-refundable Processing Fee: 1.50% to 2.50% of the gross sanctioned amount (subject to a minimum of ₹1,500 and maximum of ₹10,000) + applicable taxes.", styles['body']),
        
        Paragraph("5. Default, Penal Charges, and Repayment Operations", styles['h1']),
        Paragraph("• Penal Interest: In the event of default in payment of any installment, penal interest at the rate of <b>24% per annum (2.0% per month)</b> will be charged on overdue amounts for the duration of default.", styles['bullet']),
        Paragraph("• Cheque / NACH Dishonour Charges: ₹500 per bounce event.", styles['bullet']),
        Paragraph("• Foreclosure Statement Generation: Written foreclosure quote valid for 15 calendar days provided upon branch request or net banking initiation.", styles['bullet']),
    ]
    
    build_pdf(
        os.path.join(target_dir, "Personal_Loan_Policy.pdf"),
        "APEX TRUST BANK - PERSONAL LOAN CREDIT POLICY",
        "Document Type: Loan Policy | Reference: ATB/POL/RET-PL/2026-V3",
        [pl_p1, pl_p2]
    )

    # ==========================================
    # 3. KYC_Guidelines.pdf
    # ==========================================
    kyc_p1 = [
        Paragraph("1. Regulatory Framework & Governance", styles['h1']),
        Paragraph("These guidelines are issued in compliance with Section 35A of the Banking Regulation Act, 1949 and the Reserve Bank of India (Know Your Customer) Directions, 2016. Every branch and digital channel must rigorously apply these standards to prevent money laundering and terrorist financing.", styles['body']),
        
        Paragraph("2. The KYC Verification Process (Customer Due Diligence)", styles['h1']),
        Paragraph("Customer Due Diligence (CDD) is mandatory before opening any bank account, establishing business relationships, or sanctioning credit. The end-to-end KYC verification process comprises four critical phases:", styles['body']),
        
        Paragraph("<b>Phase 1: Officially Valid Document (OVD) Collection:</b><br/>The customer must submit proof of identity and address through any of the six prescribed OVDs: (i) Passport, (ii) Driving License, (iii) Proof of possession of Aadhaar number, (iv) Voter's Identity Card issued by Election Commission of India, (v) Job card issued by NREGA duly signed by a state government officer, or (vi) Letter issued by the National Population Register containing name and address.", styles['bullet']),
        
        Paragraph("<b>Phase 2: Permanent Account Number (PAN) Verification:</b><br/>Submission of PAN is mandatory under Rule 114B of Income Tax Rules. Real-time online validation of PAN with the NSDL/Income Tax database is executed. If PAN is unavailable, a duly attested Form 60 must be collected with proof of agricultural/non-taxable income.", styles['bullet']),
        
        Paragraph("<b>Phase 3: Customer Identification Procedure (CIP) & Physical/Digital Match:</b><br/>Verification of the customer's face, demographic particulars, and proof of current residence. Where the address in OVD differs from current residence, deemed OVDs (utility bills under 2 months, municipal tax receipt, pension payment orders) are permissible for 3 months pending address update.", styles['bullet']),
        
        Paragraph("<b>Phase 4: Video Customer Identification Process (V-CIP):</b><br/>For digital and remote account opening, bank officers conduct real-time live video calls. The process requires geo-tagging (confirming applicant is inside India), live face-match using AI matching engines against OVD photo with a confidence score above 90%, liveliness testing, OTP verification, and capture of physical PAN card under clear lighting.", styles['bullet']),
    ]
    
    kyc_p2 = [
        Paragraph("3. Risk Categorization & Periodic Updating (Re-KYC)", styles['h1']),
        Paragraph("Customers are categorized into Low, Medium, and High Risk tiers based on parameter profiles such as customer type, nature of business, geographic origin, and turnover volume:", styles['body']),
        Paragraph("<b>• High-Risk Customers:</b> Full KYC refresh must be executed every <b>2 years</b>. Includes Politically Exposed Persons (PEPs), non-resident entities, bullion dealers, jewelers, trusts, and charities.", styles['bullet']),
        Paragraph("<b>• Medium-Risk Customers:</b> Periodic KYC updating must be completed every <b>8 years</b>. Includes small business proprietors, high-volume traders, and salaried professionals with secondary business income.", styles['bullet']),
        Paragraph("<b>• Low-Risk Customers:</b> Periodic KYC updating is mandated once every <b>10 years</b>. Includes salaried employees with established corporates, government employees, pensioners, and students.", styles['bullet']),
        Paragraph("<b>• Re-KYC Self-Declaration:</b> If there is no change in KYC status for low-risk customers, a self-declaration submitted via Internet Banking, Mobile App, or registered email is accepted as compliant without visiting the branch.", styles['bullet']),
        
        Spacer(1, 8),
        Paragraph("4. Central KYC Records Registry (CKYCR)", styles['h1']),
        Paragraph("All customer KYC records are uploaded to CERSAI CKYCR within 3 days of account opening. A unique 14-digit KYC Identification Number (KIN) is generated and dispatched to the customer, enabling frictionless banking across any financial entity.", styles['body']),
    ]
    
    build_pdf(
        os.path.join(target_dir, "KYC_Guidelines.pdf"),
        "APEX TRUST BANK - MASTER KYC & AML COMPLIANCE DIRECTIVE",
        "Document Type: Compliance Guideline | Reference: ATB/COMP/KYC/2026-V1",
        [kyc_p1, kyc_p2]
    )

    # ==========================================
    # 4. RBI_Digital_Lending.pdf
    # ==========================================
    rbi_p1 = [
        Paragraph("1. Statutory Authority and Scope", styles['h1']),
        Paragraph("These directives are issued by the Reserve Bank of India pursuant to powers conferred under Sections 21, 35A and 56 of the Banking Regulation Act, 1949 and Chapter IIIB of the Reserve Bank of India Act, 1934. All Commercial Banks, Urban Co-operative Banks, and NBFCs undertaking digital lending are bound by these rules.", styles['body']),
        
        Paragraph("2. Fundamental Rules of Digital Lending", styles['h1']),
        Paragraph("The guidelines protect borrowers from predatory practices, hidden costs, illegal recovery methods, and data misuse:", styles['body']),
        
        Paragraph("<b>A. Direct Loan Disbursals and Repayments:</b><br/>All loan disbursements and repayments must be executed <b>strictly between the bank account of the borrower and the regulated entity (RE)</b>. Under no circumstances shall loan funds pass through or be routed via any pool account, escrow account, or pass-through account managed by a Lending Service Provider (LSP) or third-party fintech app.", styles['bullet']),
        
        Paragraph("<b>B. Key Fact Statement (KFS) Mandate:</b><br/>REs must provide a standardized, unambiguous Key Fact Statement (KFS) to the borrower prior to contract execution. The KFS must transparently disclose the All-Inclusive Annual Percentage Rate (APR), detailed repayment schedule, recovery charges, cooling-off provisions, and nodal grievance officer contacts. No fee or charge not mentioned in the KFS can be recovered from the borrower.", styles['bullet']),
        
        Paragraph("<b>C. Cooling-Off / Look-up Period:</b><br/>A mandatory cooling-off period must be provided during which the borrower can exit the digital loan without penalty. For loans having a tenure of <b>7 days or more</b>, the cooling-off period is a minimum of <b>3 days</b>. For loans having a tenure of less than 7 days, the cooling-off period is <b>1 day</b>. The borrower can exit by paying only the principal and proportionate APR without any foreclosure fee.", styles['bullet']),
    ]
    
    rbi_p2 = [
        Paragraph("<b>D. Digital Lending Apps (DLA) Data Privacy Restrictions:</b>", styles['h1']),
        Paragraph("Strict restrictions apply to customer data collection by DLAs and LSPs:", styles['body']),
        Paragraph("• DLAs are strictly prohibited from accessing mobile storage, contact lists, call logs, phone state, and media files.", styles['bullet']),
        Paragraph("• One-time access to camera, microphone, and location is permitted solely for customer onboarding and KYC completion, subject to prior explicit, granular consent.", styles['bullet']),
        Paragraph("• Biometric data cannot be stored on any third-party LSP server under any circumstance.", styles['bullet']),
        Paragraph("• The borrower must be provided options to revoke consent, request deletion of collected personal data, and restrict data sharing with third parties.", styles['bullet']),
        
        Spacer(1, 8),
        Paragraph("<b>E. Grievance Redressal and Nodal Officer Covenants:</b>", styles['h1']),
        Paragraph("• Regulated Entities and LSPs must appoint a dedicated Nodal Grievance Redressal Officer whose direct contact details, email, and phone number must be displayed on the website and app.", styles['body']),
        Paragraph("• Any customer complaint must be resolved within <b>30 calendar days</b> of receipt. If unresolved or rejected within 30 days, the borrower is entitled to escalate the complaint directly to the RBI Ombudsman under the Reserve Bank - Integrated Ombudsman Scheme (RB-IOS).", styles['body']),
    ]
    
    build_pdf(
        os.path.join(target_dir, "RBI_Digital_Lending.pdf"),
        "RESERVE BANK OF INDIA - REGULATORY FRAMEWORK FOR DIGITAL LENDING",
        "Document Type: Regulatory Circular | Reference: RBI/2022-23/111/DOR.CRE.REC.66",
        [rbi_p1, rbi_p2]
    )

    # ==========================================
    # 5. Insurance_Claim_Policy.pdf
    # ==========================================
    ins_p1 = [
        Paragraph("1. Purpose and Claim Philosophy", styles['h1']),
        Paragraph("Apex General & Life Insurance Corporation commits to equitable, transparent, and prompt settlement of all legitimate insurance claims. This manual articulates precise turnaround times (TAT), documentation prerequisites, and dispute handling protocols.", styles['body']),
        
        Paragraph("2. Turnaround Times (TAT) for Claim Settlement", styles['h1']),
        Paragraph("The insurance company enforces stringent operational timelines governed by IRDAI (Protection of Policyholders' Interests) Regulations:", styles['body']),
        
        Paragraph("<b>• Health Cashless Hospitalization Claims:</b><br/>Initial cashless authorization approval or denial must be communicated within <b>1 hour (60 minutes)</b> of receipt of the pre-authorization request from network hospital. Final cashless discharge authorization must be finalized within <b>2 hours</b> of receiving complete final billing and discharge summary.", styles['bullet']),
        
        Paragraph("<b>• Health Reimbursement Claims:</b><br/>Admissible health reimbursement claims must be settled or rejected within <b>15 calendar days</b> from the date of receipt of the last necessary document from the claimant.", styles['bullet']),
        
        Paragraph("<b>• Life Insurance Non-Early Death Claims:</b><br/>For death claims arising on policies active for greater than 3 continuous years (where Section 45 applies), the entire claim settlement must be executed within <b>15 days</b> from the date of submission of all required claim papers.", styles['bullet']),
        
        Paragraph("<b>• Life Insurance Early Death Claims (Requiring Investigation):</b><br/>Where death occurs within 3 years of policy inception and investigation is necessitated to rule out non-disclosure or fraud: The investigation must be completed within <b>90 days</b> of claim intimation. The final claim settlement or repudiation must be concluded within <b>30 days</b> of investigation completion (maximum total TAT not exceeding 120 days).", styles['bullet']),
        
        Paragraph("<b>• Delayed Settlement Penal Interest:</b><br/>In case of any unexcused delay beyond the prescribed 15/30 day window, the insurer is legally liable to pay penal interest to the beneficiary at a rate of <b>2% above the prevailing bank rate</b> calculated from the date of receipt of the last document until actual disbursement.", styles['bullet']),
    ]
    
    ins_p2 = [
        Paragraph("3. Mandatory Claim Documentation Checklist", styles['h1']),
        Paragraph("Claimants must furnish authentic documentation according to the claim category:", styles['body']),
        Paragraph("• Health Claims: Original detailed hospital discharge card, itemized final bill with payment receipts, diagnostic and lab test reports with prescriptions, doctor consultation notes, pharmacy purchase invoices, and indoor case papers (for surgical admissions).", styles['bullet']),
        Paragraph("• Life Death Claims: Duly executed death claim form, original death certificate issued by Municipal Corporation / Gram Panchayat, original policy document, claimant KYC (PAN and Aadhaar), cancelled cheque for NEFT credit, and Attending Physician's Medical Statement.", styles['bullet']),
        Paragraph("• Accidental Death Claims: In addition to standard death documents, submit certified copies of First Information Report (FIR), Police Inquest Panchnama, Post-Mortem Examination Report, and Medico-Legal Certificate (MLC).", styles['bullet']),
        
        Spacer(1, 8),
        Paragraph("4. Grievance Escalation & Insurance Ombudsman", styles['h1']),
        Paragraph("If a claim is rejected or unresolved by the insurer's Grievance Redressal Officer within 15 days, the policyholder can register a grievance on the IRDAI Bima Bharosa portal or escalate the matter to the territorial Insurance Ombudsman within 1 year.", styles['body']),
    ]
    
    build_pdf(
        os.path.join(target_dir, "Insurance_Claim_Policy.pdf"),
        "APEX INSURANCE - COMPREHENSIVE CLAIM SETTLEMENT GUIDELINES",
        "Document Type: Insurance Policy | Reference: AIC/POL/CLM-TAT/2026",
        [ins_p1, ins_p2]
    )

    # ==========================================
    # 6. Credit_Card_Terms.pdf
    # ==========================================
    cc_p1 = [
        Paragraph("1. Cardholder Agreement and Scope", styles['h1']),
        Paragraph("This document sets forth the Most Important Terms and Conditions (MITC) governing credit card accounts issued by Apex Trust Bank. Acceptance of the physical or virtual credit card constitutes full agreement with these covenants.", styles['body']),
        
        Paragraph("2. Billing Cycle, Interest-Free Period, and APR", styles['h1']),
        Paragraph("<b>• Billing Cycle:</b> Statements are generated monthly on a recurring fixed billing date covering a 30 to 31 day transaction period.", styles['bullet']),
        Paragraph("<b>• Interest-Free Grace Period:</b> Ranges between <b>20 to 50 days</b> depending upon transaction date within the cycle. Valid only if the previous statement Total Amount Due was paid in full on or prior to the payment due date.", styles['bullet']),
        Paragraph("<b>• Annual Percentage Rate (APR) and Finance Charges:</b> For retail revolving balances and cash advances, interest is levied at <b>3.65% per month (43.8% p.a.)</b> computed on average daily balance from transaction date until repayment.", styles['bullet']),
        Paragraph("<b>• Cash Advance Transaction Fee:</b> 2.5% of the cash withdrawn or ₹500 (whichever is higher), with finance charge accruing immediately from the day of withdrawal.", styles['bullet']),
        
        Spacer(1, 8),
        Paragraph("3. Minimum Amount Due (MAD) and Late Charges", styles['h1']),
        Paragraph("<b>• Minimum Amount Due Computation:</b> Calculated as <b>5% of Total Outstanding Balance + EMI installments + any unpaid MAD from prior cycles + applicable fees/taxes</b>.", styles['bullet']),
        Paragraph("<b>• Late Payment Fee Structure:</b><br/>• Statement balance up to ₹1,000: Nil<br/>• Balance ₹1,001 to ₹5,000: ₹500<br/>• Balance ₹5,001 to ₹10,000: ₹750<br/>• Balance exceeding ₹10,000: ₹1,200.", styles['bullet']),
    ]
    
    cc_p2 = [
        Paragraph("4. Zero Liability on Fraudulent Transactions & Chargebacks", styles['h1']),
        Paragraph("• Fraud Notification Window: If an unauthorized transaction occurs, cardholder liability is <b>Zero</b> if reported within <b>3 calendar days</b> of unauthorized transaction notice.", styles['bullet']),
        Paragraph("• If reported between 4 to 7 calendar days, customer liability is limited to the lower of transaction amount or ₹10,000. Delays beyond 7 days are subject to bank board-approved discretionary policy.", styles['bullet']),
        Paragraph("• Chargeback Dispute Window: Chargebacks for merchant non-delivery or billing errors must be lodged within <b>120 calendar days</b> of the transaction date.", styles['bullet']),
        
        Paragraph("5. Card Cancellation and Closure Terms", styles['h1']),
        Paragraph("Cardholders can initiate card closure at any time via phone banking, app, or email. The bank will permanently cancel the card within 7 business days, provided outstanding balance is fully settled.", styles['body']),
    ]
    
    build_pdf(
        os.path.join(target_dir, "Credit_Card_Terms.pdf"),
        "APEX TRUST BANK - CREDIT CARD CARDHOLDER TERMS & TARIFF",
        "Document Type: Product Terms | Reference: ATB/CRD/MITC/2026-V2",
        [cc_p1, cc_p2]
    )

    # ==========================================
    # 7. Banking_SOP.pdf
    # ==========================================
    sop_p1 = [
        Paragraph("1. Purpose and Standard Operating Protocol", styles['h1']),
        Paragraph("This manual specifies standard branch operational procedures (SOP) across Apex Trust Bank branches, covering cash governance, vault protocol, dormant account lifecycle, and deceased claim settlement.", styles['body']),
        
        Paragraph("2. Cash Management & Branch Vault Protocol", styles['h1']),
        Paragraph("<b>• Dual Custody:</b> The strong room and vault must be operated under dual custody by the Branch Operations Manager (BOM) and Head Cashier at all times.", styles['bullet']),
        Paragraph("<b>• High-Value Cash Transaction Rules:</b> Cash deposits exceeding ₹50,000 necessitate mandatory PAN quotation. Cash transactions of ₹10,00,000 and above trigger mandatory source of wealth declaration and automated filing of Cash Transaction Report (CTR) with FIU-IND.", styles['bullet']),
        Paragraph("<b>• Daily Cash Limit & Retention:</b> Any cash holding in branch vault exceeding the sanctioned retention limit at end of day must be remitted to the Currency Chest by 16:30 hrs.", styles['bullet']),
        
        Spacer(1, 8),
        Paragraph("3. Inoperative & Dormant Account Reactivation SOP", styles['h1']),
        Paragraph("<b>• Inoperative Classification:</b> A savings or current bank account is classified as inoperative / dormant if there are no customer-induced financial transactions for a continuous period of <b>24 months (2 years)</b>.", styles['bullet']),
        Paragraph("<b>• Reactivation Requirements:</b> To reactivate a dormant account, the account holder must present in-person with fresh KYC documents (OVD + PAN + recent photograph). Branch staff must verify signatures and complete biometric/face verification.", styles['bullet']),
        Paragraph("<b>• Zero Reactivation Fee:</b> In accordance with RBI directives, <b>no fee or penalty can be levied</b> for reactivating dormant or inoperative accounts.", styles['bullet']),
    ]
    
    sop_p2 = [
        Paragraph("4. Deceased Depositor Claim Settlement SOP", styles['h1']),
        Paragraph("<b>• Accounts with Valid Nomination:</b> Claims with valid nominee registration must be settled within <b>15 calendar days</b> of receiving the official death certificate and KYC proof of the nominee, without demanding succession certificate, probate, or letters of administration.", styles['bullet']),
        Paragraph("<b>• Accounts without Nomination (Up to ₹5,00,000):</b> Settlement is executed in favor of legal heirs against an Indemnity Bond with two independent guarantors, death certificate, and Legal Heirship certificate.", styles['bullet']),
        
        Spacer(1, 8),
        Paragraph("5. Safe Deposit Locker Operation Guidelines", styles['h1']),
        Paragraph("• Access to lockers is permitted only after biometric/signature verification against branch master register.", styles['bullet']),
        Paragraph("• Under the revised RBI locker guidelines, the bank's maximum liability for locker contents in instances of burglary, theft, building collapse, or staff fraud is capped at <b>100 times the annual locker rent</b>.", styles['bullet']),
    ]
    
    build_pdf(
        os.path.join(target_dir, "Banking_SOP.pdf"),
        "APEX TRUST BANK - STANDARD OPERATING PROCEDURES (SOP)",
        "Document Type: Banking SOP | Reference: ATB/SOP/OPS/2026-V5",
        [sop_p1, sop_p2]
    )

    # ==========================================
    # 8. Finance_FAQ.pdf
    # ==========================================
    faq_p1 = [
        Paragraph("1. Home Loan Inquiries", styles['h1']),
        Paragraph("<b>Q: What documents are required for home loan?</b><br/>A: Applicants must provide (1) KYC Proofs (PAN Card mandatory, plus Aadhaar/Passport/Voter ID), (2) Income Proof (for salaried: last 3 months' pay slips, Form 16 for past 2 years, 6 months' salary bank statements; for self-employed: 3 years' audited ITR, Balance Sheets, and 12 months' business bank statements), and (3) Property Documents (registered sale agreement, approved building plan, 30-year Encumbrance Certificate, and builder/society NOC).", styles['body']),
        Paragraph("<b>Q: What is the minimum age for home loan?</b><br/>A: The minimum age for a home loan applicant is <b>21 years</b> at the time of application, with a maximum age cap of 65 years for salaried and 70 years for self-employed at maturity.", styles['body']),
        
        Spacer(1, 6),
        Paragraph("2. Personal Loan Inquiries", styles['h1']),
        Paragraph("<b>Q: What is the foreclosure charge for personal loan?</b><br/>A: Personal loans cannot be foreclosed during the first 6 months lock-in period. Thereafter, foreclosure charges are structured as: <b>4% of outstanding principal</b> (months 7 to 12), <b>3%</b> (months 13 to 24), <b>2%</b> (months 25 to 36), and <b>1%</b> (beyond 36 months) + applicable GST.", styles['body']),
        
        Spacer(1, 6),
        Paragraph("3. Regulatory & Digital Lending Inquiries", styles['h1']),
        Paragraph("<b>Q: What are RBI digital lending rules?</b><br/>A: Core RBI digital lending rules mandate: (1) Direct loan disbursal and repayment strictly between the borrower's and the bank's account with zero pass-through accounts, (2) Mandatory Key Fact Statement (KFS) stating all-inclusive APR, (3) Minimum 3-day cooling-off period for loans of 7+ days, (4) Zero access by digital lending apps to contacts, phone storage, or call logs, and (5) Customer grievance redressal within 30 days.", styles['body']),
    ]
    
    faq_p2 = [
        Paragraph("4. Insurance Claim Inquiries", styles['h1']),
        Paragraph("<b>Q: How many days does insurance claim settlement take?</b><br/>A: Health cashless claims take <b>1 hour</b> for initial authorization and <b>2 hours</b> for final discharge approval. Health reimbursement claims must be settled within <b>15 days</b> of receiving all documents. Life insurance non-early death claims are settled within <b>15 days</b>. Early claims under investigation take up to 90 days for investigation and must be settled within <b>30 days</b> thereafter.", styles['body']),
        
        Spacer(1, 6),
        Paragraph("5. KYC Compliance Inquiries", styles['h1']),
        Paragraph("<b>Q: What is KYC verification process?</b><br/>A: The KYC verification process involves 4 key steps: (1) Collection of Officially Valid Document (Passport, Driving License, Aadhaar, Voter ID, NREGA job card, or NPR letter), (2) Online verification of PAN card with Income Tax database, (3) Customer Identification Procedure (CIP) with current address verification, and (4) Video Customer Identification Process (V-CIP) with live geo-tagging, face-match (>90%), and liveliness test.", styles['body']),
        
        Spacer(1, 6),
        Paragraph("6. Deposits & Tax Deducted at Source (TDS)", styles['h1']),
        Paragraph("<b>Q: What is the TDS threshold on Fixed Deposit interest?</b><br/>A: Under Section 194A of Income Tax Act, TDS is deducted if interest income exceeds ₹40,000 per financial year for individuals under 60 years, or ₹50,000 for senior citizens (60+ years). To prevent TDS deduction, eligible depositors can submit Form 15G (for general citizens) or Form 15H (for senior citizens) at the beginning of each financial year.", styles['body']),
    ]
    
    build_pdf(
        os.path.join(target_dir, "Finance_FAQ.pdf"),
        "APEX FINANCIAL CONSUMER FAQ & ADVISORY COMPENDIUM",
        "Document Type: FAQ Document | Reference: ATB/FAQ/GEN/2026",
        [faq_p1, faq_p2]
    )
    
    print(f"Successfully generated all 8 sample financial PDFs in '{target_dir}' directory.")


if __name__ == "__main__":
    generate_all_sample_pdfs("documents")
