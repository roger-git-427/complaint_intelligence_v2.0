"""Canonical CFPB column names. Bronze may use CSV or API labels; silver is snake_case."""

COLUMN_RENAME = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "complaint_what_happened": "narrative",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Consumer consent provided?": "consumer_consent",
    "consumer_consent_provided": "consumer_consent",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
    "timely": "timely_response",
    "Consumer disputed?": "consumer_disputed",
}

SILVER_COLUMNS = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "narrative",
    "company_public_response",
    "company",
    "state",
    "zip_code",
    "tags",
    "consumer_consent",
    "submitted_via",
    "date_sent_to_company",
    "company_response",
    "timely_response",
    "consumer_disputed",
]

NULL_TOKENS = {"", "n/a", "na", "none", "null", "nan", "none provided"}
