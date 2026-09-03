"""Shared tiny CFPB-like frames for unit tests."""

from __future__ import annotations

import pandas as pd


def sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "complaint_id": "1001",
                "date_received": "2024-06-01",
                "product": "Credit card",
                "sub_product": "General-purpose credit card",
                "issue": "Fees or interest",
                "sub_issue": "",
                "complaint_what_happened": "Bank charged an unexplained late fee on my credit card after autopay failed.",
                "company_public_response": "",
                "company": "Example Bank NA",
                "state": "TX",
                "zip_code": "78701",
                "tags": "",
                "consumer_consent_provided": "Consent provided",
                "submitted_via": "Web",
                "date_sent_to_company": "2024-06-02",
                "company_response": "Closed with explanation",
                "timely": "Yes",
                "consumer_disputed": "No",
            },
            {
                "complaint_id": "1001",  # duplicate — silver keeps last
                "date_received": "2024-06-03",
                "product": "Credit card",
                "sub_product": "General-purpose credit card",
                "issue": "Fees or interest",
                "sub_issue": "",
                "complaint_what_happened": "Updated: Bank charged an unexplained late fee on my credit card after autopay failed.",
                "company_public_response": "",
                "company": "Example Bank NA",
                "state": "TX",
                "zip_code": "78701",
                "tags": "",
                "consumer_consent_provided": "Consent provided",
                "submitted_via": "Web",
                "date_sent_to_company": "2024-06-04",
                "company_response": "Closed with explanation",
                "timely": "Yes",
                "consumer_disputed": "No",
            },
            {
                "complaint_id": "1002",
                "date_received": "2024-07-15",
                "product": "Mortgage",
                "sub_product": "Conventional home mortgage",
                "issue": "Trouble during payment process",
                "sub_issue": "",
                "complaint_what_happened": "Mortgage servicer lost my payment and threatened foreclosure despite proof of wire transfer.",
                "company_public_response": "",
                "company": "Sample Mortgage Co",
                "state": "CA",
                "zip_code": "94105",
                "tags": "",
                "consumer_consent_provided": "Consent provided",
                "submitted_via": "Web",
                "date_sent_to_company": "2024-07-16",
                "company_response": "Closed with monetary relief",
                "timely": "No",
                "consumer_disputed": "Yes",
            },
        ]
    )
