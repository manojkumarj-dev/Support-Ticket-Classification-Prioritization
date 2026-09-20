"""Dataset generation script for Support Ticket Classification & Prioritization.

Generates a realistic corpus of 2,500 highly diverse support tickets across 5 categories
and 4 priority levels, with varied lengths, real-world error traces, user frustration cues,
natural typos, and domain vocabulary overlap.
"""

import random
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

random.seed(config.RANDOM_STATE)

OPENINGS = [
    "Customer reports:", "User stated:", "Ticket opened by employee:", "Escalation notice:",
    "Automated monitoring alert:", "Urgent assistance requested:", "Support inquiry from tier-1:",
    "Report from customer success:", "Internal incident report:", "Client wrote:",
    "Issue logged via web portal:", "Direct message from enterprise admin:", "Inquiry received:",
    "Helpdesk dispatch:", "Critical alert logged:"
]

ENVIRONMENTS = [
    "on production cluster us-east-1", "in staging environment", "on employee MacBook Pro (macOS Sonoma)",
    "using Chrome 122 on Windows 11", "on Firefox ESR 115", "in internal Kubernetes namespace 'billing-prod'",
    "on mobile client iOS 17.4", "via Android app v4.2.1", "on corporate Lenovo ThinkPad T14",
    "connecting through corporate Zscaler VPN", "on dedicated AWS EC2 c5.4xlarge instance",
    "within the European data region eu-west-1", "on Ubuntu 22.04 LTS workstation"
]

IMPACTS = [
    "This is blocking the entire finance department from closing month-end accounts.",
    "Causes immediate revenue loss and prevents checkout completion.",
    "Halts all production CI/CD deployments across engineering.",
    "Affects more than 2,000 active concurrent users.",
    "Only affects a single user, but workaround is cumbersome.",
    "No immediate workaround available; critical customer escalation.",
    "Visual bug only; functional workflows remain operational.",
    "Minor inconvenience for internal staff.",
    "High severity as our SLA requires a resolution within 2 hours.",
    "Low impact, customer is just inquiring for future planning."
]

ERRORS = [
    "HTTP 500 Internal Server Error", "HTTP 502 Bad Gateway", "HTTP 504 Gateway Timeout",
    "HTTP 403 Forbidden - Access Denied", "HTTP 401 Unauthorized", "java.lang.NullPointerException",
    "Database connection pool exhausted", "Deadlock detected in transaction #TX-9482",
    "OutOfMemoryError: Java heap space", "SSL handshake failed: certificate expired",
    "ERR_CONNECTION_REFUSED on port 8080", "SocketTimeoutException after 30000ms"
]

CATEGORY_DATA = {
    "Bug / System Error": {
        "templates": [
            "Application crashes consistently when {action}. Error message: '{error}'. Environment: {env}. {impact}",
            "Encountered a fatal crash during {action}. The screen went completely blank and logged {error}. {impact}",
            "Bug discovered in {system}: {action} yields incorrect calculation results. {impact}",
            "Unhandled exception thrown during {action}. Stack trace indicates {error}. {env}. {impact}",
            "Data corruption risk: After {action}, previously saved records disappear from {system}. {impact}",
            "Intermittent failure: {action} fails with {error} for about 15% of requests. {impact}",
            "Infinite loop / UI freeze occurs whenever a user attempts {action} on {env}. {impact}",
            "The export feature is generating corrupted, 0-byte files when {action}. {impact}",
            "Search index seems desynchronized in {system}; new entries do not appear after {action}. {impact}",
            "Crash report: Worker thread terminated unexpectedly with {error} while {action}. {impact}",
            "UI regression: Form fields overlap and become unclickable in {system} when {action}. {impact}",
            "Typo and visual bug: Text label displays 'Unkown Error' instead of valid status on {system}."
        ],
        "actions": [
            "generating the annual revenue reconciliation report", "uploading a 25MB CSV data file",
            "clicking the bulk submit button", "filtering transaction logs by timestamp",
            "rendering the executive KPI dashboard", "invoking the batch synchronization API",
            "processing payroll deductions for all employees", "saving nested profile permissions",
            "importing customer leads from Salesforce", "executing SQL query against reporting replica"
        ],
        "systems": ["Reporting Engine", "Analytics Dashboard", "Checkout Service", "Batch Processing Worker",
                    "Ingestion Pipeline", "Inventory Microservice", "Search Backend", "Notification Queue"],
        "priorities": ["Critical", "High", "Critical", "High", "Critical", "Medium", "High", "Medium", "Medium", "Critical", "Low", "Low"]
    },
    "Billing & Payment": {
        "templates": [
            "Our corporate account was billed twice for invoice #{num}. Total overcharge is ${amount}. {impact}",
            "Credit card payment failed with code ERR_CARD_PROCESSOR_TIMEOUT for company {org}. {impact}",
            "Cannot download tax / VAT invoice receipt for invoice #{num}. Need it immediately for external audit. {impact}",
            "Please update our billing address to {address} and re-send the invoice #{num}. {impact}",
            "Disputed transaction: We were charged the Enterprise annual tier rate of ${amount} instead of monthly tier. {impact}",
            "Account marked past due and suspended, but wire transfer of ${amount} was completed 3 business days ago. {impact}",
            "Requesting official formal price quote for expanding from 50 to 150 user licenses for {org}. {impact}",
            "Please remove expired corporate credit card ending in {card} and assign the new Amex as default. {impact}",
            "Subscription auto-renewed without prior 30-day notice; requesting full refund for unused seat licenses. {impact}",
            "Payment gateway webhook failed after customer checkout; customer was charged but order shows unpaid. {impact}",
            "Need clarification on line-item usage charges for data egress on current monthly billing statement. {impact}",
            "Invoice #{num} has the incorrect legal entity tax ID. Please regenerate with GST/VAT #VAT-{num}. {impact}"
        ],
        "actions": ["processing monthly subscription", "updating payment gateway", "downloading receipts"],
        "systems": ["Billing Portal", "Stripe Gateway", "Invoice System", "Subscription Manager"],
        "priorities": ["High", "High", "Medium", "Low", "High", "Critical", "Low", "Low", "Medium", "Critical", "Low", "Medium"]
    },
    "Account & Access": {
        "templates": [
            "Locked out of account. Password reset email never arrives for user {email}. {impact}",
            "Two-Factor Authentication (2FA) SMS verification code is not being delivered to phone ending in {card}. {impact}",
            "Company-wide SSO login outage: Okta SAML returns 'Invalid assertion signature'. All employees locked out! {impact}",
            "Please grant Administrator privileges and workspace owner role to new hire {email}. {impact}",
            "User session randomly terminates every 3 minutes, forcing re-authentication on {env}. {impact}",
            "Immediate offboarding request: Please revoke all API tokens and deactivate account for former staff {email}. {impact}",
            "Password reset link gives 'HTTP 400 Token Invalid or Expired' immediately upon opening. {impact}",
            "Received '403 Forbidden: Insufficient Permissions' when trying to access workspace '{org}' despite being admin. {impact}",
            "Account locked after multiple failed attempts. Please unlock username '{user}' immediately. {impact}",
            "Need help configuring Role-Based Access Control (RBAC) permissions for third-party auditing contractors. {impact}",
            "How can I transfer primary account ownership from {email} to finance@company.com? {impact}",
            "Security escalation: Unusual login detected from unrecognized IP in foreign jurisdiction for account {user}. {impact}"
        ],
        "actions": ["logging into portal", "authenticating with 2FA", "resetting password"],
        "systems": ["Auth Service", "Okta SSO", "Identity Provider", "RBAC Engine"],
        "priorities": ["High", "Critical", "Critical", "Medium", "Medium", "High", "High", "Medium", "Medium", "Low", "Low", "Critical"]
    },
    "Feature Request": {
        "templates": [
            "Feature Request: Please add native Dark Mode support across the entire web application dashboard. {impact}",
            "We would love the ability to export all chart visualizations directly to high-res SVG or vector PDF. {impact}",
            "Requesting Webhook support for ticket status updates so our engineering team can trigger custom Slack alerts. {impact}",
            "Can you support bulk employee CSV upload? Manually creating individual user accounts is too slow. {impact}",
            "Feature enhancement: Support multi-currency checkout (EUR, GBP, JPY, CAD) on the customer-facing billing page. {impact}",
            "Would appreciate an automated weekly email digest summarizing team performance and open ticket counts. {impact}",
            "Please introduce keyboard navigation shortcuts (e.g., 'j/k' to traverse items, 'e' to archive) for power users. {impact}",
            "Requesting comprehensive audit log viewer in the admin console to track configuration modifications. {impact}",
            "Feature request: Allow custom branding and company logo white-labeling on client-facing export reports. {impact}",
            "Can we add granular tag-based filtering and custom multi-select fields to the main ticket view? {impact}",
            "Please provide a public REST API endpoint allowing programmatic retrieval of workspace analytics metrics. {impact}",
            "Kindly increase maximum file upload limit from 10MB to 100MB to support uploading short debugging screen recordings. {impact}"
        ],
        "actions": ["navigating dashboard", "customizing workspace", "managing workflows"],
        "systems": ["Web UI", "Settings Panel", "Export Module", "Reporting Service"],
        "priorities": ["Low", "Low", "Medium", "Medium", "Medium", "Low", "Low", "Medium", "Low", "Low", "Medium", "Low"]
    },
    "Technical / IT Support": {
        "templates": [
            "Corporate VPN gateway is rejecting connections with 'TLS Handshake Failed'. Remote engineers cannot access servers. {impact}",
            "Office Wi-Fi in the 4th floor engineering wing is suffering severe packet loss and DNS resolution timeouts. {impact}",
            "Dual external monitors are not recognized when connecting Lenovo ThinkPad to the USB-C docking station. {impact}",
            "Outlook desktop app crashes repeatedly during startup with error 'MAPI32.DLL missing or corrupt'. {impact}",
            "Requesting standard workstation hardware provisioning for incoming senior software engineer starting next week. {impact}",
            "Floor 3 network printer is displaying 'Paper Jam Error 52' even though the internal tray is completely clear. {impact}",
            "Local Docker desktop engine fails to start, reporting 'Hardware-assisted virtualization is disabled in UEFI'. {impact}",
            "Internal code repository server (git.internal.corp) is completely unreachable; SSH connection timed out on port 22. {impact}",
            "Need replacement 90W USB-C AC power adapter for Dell Latitude laptop; current cable has a frayed wire. {impact}",
            "CrowdStrike endpoint sensor is consuming 98% CPU continuously, causing laptop to freeze. {impact}",
            "Assistance needed to configure corporate AWS CLI credentials and kubectl context on developer workstation. {impact}",
            "Physical Ethernet port at desk #38 has no link light and does not assign a DHCP IP address. {impact}"
        ],
        "actions": ["connecting to network", "launching local tools", "configuring hardware"],
        "systems": ["Corporate VPN", "Office Wi-Fi", "Developer Laptop", "Internal Git Server"],
        "priorities": ["Critical", "High", "Low", "Medium", "Low", "Low", "Medium", "Critical", "Low", "High", "Low", "Low"]
    }
}

ORGS = ["Acme Corp", "Globex Global", "Stark Industries", "Wayne Enterprises", "Initech LLC",
        "Umbrella Corp", "Cyberdyne Systems", "Hooli Tech", "Pied Piper", "Massive Dynamic"]
USERS = ["j.smith", "a.patel", "m.garcia", "k.chen", "d.rossi", "l.muller", "t.tanaka", "e.dubois"]
CITIES = ["New York", "London", "San Francisco", "Tokyo", "Berlin", "Singapore", "Sydney", "Toronto"]

def generate_ticket(category: str, ticket_id: int) -> dict:
    cat_info = CATEGORY_DATA[category]
    idx = random.randint(0, len(cat_info["templates"]) - 1)
    template = cat_info["templates"][idx]
    base_priority = cat_info["priorities"][idx]
    
    org = random.choice(ORGS)
    user = random.choice(USERS)
    email = f"{user}@{org.lower().replace(' ', '')}.com"
    address = f"{random.randint(100, 999)} Market St, {random.choice(CITIES)}"
    
    text = template.format(
        action=random.choice(cat_info["actions"]),
        system=random.choice(cat_info["systems"]),
        error=random.choice(ERRORS),
        env=random.choice(ENVIRONMENTS),
        impact=random.choice(IMPACTS),
        num=random.randint(10000, 99999),
        amount=f"{random.randint(50, 4500):,}",
        card=random.randint(1000, 9999),
        org=org,
        user=user,
        email=email,
        address=address
    )
    
    # 60% chance to prefix with realistic opening
    if random.random() < 0.60:
        text = f"{random.choice(OPENINGS)} {text}"
        
    # 15% chance of slight priority shift (human triage subjectivity)
    priority = base_priority
    if random.random() < 0.15:
        priority = random.choice(config.PRIORITIES)
        
    return {
        "ticket_id": f"TCK-{ticket_id:05d}",
        "text": text,
        "category": category,
        "priority": priority
    }


def generate_dataset(output_path: Path = config.RAW_DATA_PATH, total_samples: int = 2500) -> None:
    categories = list(CATEGORY_DATA.keys())
    samples_per_cat = total_samples // len(categories)
    
    records = []
    ticket_id = 10001
    
    for category in categories:
        for _ in range(samples_per_cat):
            records.append(generate_ticket(category, ticket_id))
            ticket_id += 1
            
    random.shuffle(records)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "text", "category", "priority"])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Generated {len(records)} tickets at: {output_path}")


if __name__ == "__main__":
    generate_dataset()
