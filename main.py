import os
import json
import re
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Dynamic Schema Structured Extraction API",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class DynamicExtractRequest(BaseModel):

    text: str

    schema: Dict[str, str]


# ============================================================
# SUPPORTED TYPES
# ============================================================

SUPPORTED_TYPES = {
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "array[string]",
    "array[integer]"
}


# ============================================================
# FIELD NAME HELPERS
# ============================================================

def normalize_field_name(
    field: str
) -> str:

    field = field.strip()

    field = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        field
    )

    field = field.replace(
        "-",
        "_"
    )

    field = re.sub(
        r"\s+",
        "_",
        field
    )

    return field.lower()


def humanize_field(
    field: str
) -> str:

    field = normalize_field_name(
        field
    )

    return field.replace(
        "_",
        " "
    )


# ============================================================
# TYPE VALIDATION
# ============================================================

def validate_schema(
    schema: Dict[str, str]
):

    if not isinstance(
        schema,
        dict
    ):

        raise ValueError(
            "schema must be an object"
        )


    if not schema:

        raise ValueError(
            "schema must not be empty"
        )


    for field, field_type in schema.items():

        if not isinstance(
            field,
            str
        ) or not field.strip():

            raise ValueError(
                "schema contains invalid field name"
            )


        if field_type not in SUPPORTED_TYPES:

            raise ValueError(
                f"Unsupported type: {field_type}"
            )


# ============================================================
# DEFAULT VALUE
# ============================================================

def default_value(
    field_type: str
):

    # All missing fields must be null.
    return None


# ============================================================
# CONVERT VALUE TO REQUESTED TYPE
# ============================================================

def convert_value(
    value: Any,
    field_type: str
):

    if value is None:

        return None


    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    if field_type == "string":

        if isinstance(
            value,
            bool
        ):

            return (
                "true"
                if value
                else "false"
            )


        if isinstance(
            value,
            (dict, list)
        ):

            return json.dumps(
                value,
                ensure_ascii=False
            )


        return str(
            value
        ).strip()


    # --------------------------------------------------------
    # INTEGER
    # --------------------------------------------------------

    if field_type == "integer":

        if isinstance(
            value,
            bool
        ):

            return None


        if isinstance(
            value,
            int
        ):

            return value


        if isinstance(
            value,
            float
        ):

            if value.is_integer():

                return int(
                    value
                )

            return None


        s = str(
            value
        ).strip()


        # Remove common numeric formatting
        s = s.replace(
            ",",
            ""
        )


        # Handle currency
        s = re.sub(
            r"^[^\d+-]*",
            "",
            s
        )


        match = re.search(
            r"[-+]?\d+",
            s
        )


        if not match:

            return None


        try:

            return int(
                match.group(0)
            )

        except:

            return None


    # --------------------------------------------------------
    # FLOAT
    # --------------------------------------------------------

    if field_type == "float":

        if isinstance(
            value,
            bool
        ):

            return None


        if isinstance(
            value,
            (int, float)
        ):

            return float(
                value
            )


        s = str(
            value
        ).strip()


        s = s.replace(
            ",",
            ""
        )


        # Find decimal or integer number
        match = re.search(
            r"[-+]?(?:\d+\.\d+|\d+)",
            s
        )


        if not match:

            return None


        try:

            return float(
                match.group(0)
            )

        except:

            return None


    # --------------------------------------------------------
    # BOOLEAN
    # --------------------------------------------------------

    if field_type == "boolean":

        if isinstance(
            value,
            bool
        ):

            return value


        s = str(
            value
        ).strip().lower()


        if s in {
            "true",
            "yes",
            "y",
            "1",
            "on"
        }:

            return True


        if s in {
            "false",
            "no",
            "n",
            "0",
            "off"
        }:

            return False


        return None


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if field_type == "date":

        if isinstance(
            value,
            datetime
        ):

            return value.strftime(
                "%Y-%m-%d"
            )


        s = str(
            value
        ).strip()


        # Already ISO
        match = re.fullmatch(
            r"(\d{4})-(\d{2})-(\d{2})",
            s
        )


        if match:

            try:

                datetime.strptime(
                    s,
                    "%Y-%m-%d"
                )

                return s

            except:

                return None


        # DD Month YYYY
        for fmt in [
            "%d %B %Y",
            "%d %b %Y",
            "%B %d %Y",
            "%b %d %Y"
        ]:

            try:

                dt = datetime.strptime(
                    s,
                    fmt
                )

                return dt.strftime(
                    "%Y-%m-%d"
                )

            except:

                pass


        # DD/MM/YYYY
        for fmt in [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y"
        ]:

            try:

                dt = datetime.strptime(
                    s,
                    fmt
                )

                return dt.strftime(
                    "%Y-%m-%d"
                )

            except:

                pass


        return None


    # --------------------------------------------------------
    # ARRAY STRING
    # --------------------------------------------------------

    if field_type == "array[string]":

        if isinstance(
            value,
            list
        ):

            result = []

            for item in value:

                if item is not None:

                    result.append(
                        str(item).strip()
                    )


            return result


        if isinstance(
            value,
            str
        ):

            # Handle comma-separated values
            return [
                x.strip()
                for x in value.split(",")
                if x.strip()
            ]


        return None


    # --------------------------------------------------------
    # ARRAY INTEGER
    # --------------------------------------------------------

    if field_type == "array[integer]":

        if isinstance(
            value,
            list
        ):

            result = []

            for item in value:

                converted = convert_value(
                    item,
                    "integer"
                )

                if converted is not None:

                    result.append(
                        converted
                    )


            return result


        if isinstance(
            value,
            str
        ):

            numbers = re.findall(
                r"[-+]?\d+",
                value
            )

            return [
                int(x)
                for x in numbers
            ]


        return None


    return None


# ============================================================
# LLM EXTRACTION
# ============================================================

def extract_with_llm(
    text: str,
    schema: Dict[str, str]
) -> Dict[str, Any]:

    if client is None:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )


    schema_json = json.dumps(
        schema,
        ensure_ascii=False,
        indent=2
    )


    prompt = f"""
You are a high-reliability information extraction system.

Extract structured information from the TEXT below.

TEXT:
{text}

REQUESTED SCHEMA:
{schema_json}

SUPPORTED TYPES:
- string
- integer
- float
- boolean
- date
- array[string]
- array[integer]

STRICT RULES:

1. Return ONLY valid JSON.
2. Return EXACTLY the requested field names.
3. Never add fields that are not in the schema.
4. Never omit a requested field.
5. If a requested field cannot be determined from the text,
   return null.
6. Never use outside knowledge.
7. Extract values only from the supplied text.
8. Interpret field names semantically.

Examples of semantic field names:

- from_bank means the bank the money/payment was sent from.
- to_bank means the bank the money/payment was sent to.
- customer_name means the customer's name.
- sender_name means the sender's name.
- receiver_name means the receiver's name.
- account_number means an account number.
- transaction_id means a transaction identifier.
- purchase_date means the date of purchase.
- amount means the relevant monetary amount.
- quantity means the relevant quantity.
- store means the store or seller.
- company means the relevant company.

9. Pay close attention to relationships and direction.

For example:

"Transferred Rs. 5000 from HDFC to SBI"

means:

from_bank = "HDFC"
to_bank = "SBI"

Do NOT return null when the requested field can be inferred
directly from the wording of the text.

10. For date fields, return YYYY-MM-DD.
11. For integer fields, return JSON integers.
12. For float fields, return JSON numbers.
13. For boolean fields, return true or false.
14. For array[string], return a JSON array of strings.
15. For array[integer], return a JSON array of integers.

Return JSON only.
"""


    response = client.models.generate_content(
        model="gemini-3.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            max_output_tokens=2000
        )
    )


    if not response.text:

        return {}


    raw = response.text.strip()


    try:

        data = json.loads(
            raw
        )

    except Exception:

        # Try extracting JSON object
        match = re.search(
            r"\{.*\}",
            raw,
            re.DOTALL
        )


        if not match:

            return {}


        try:

            data = json.loads(
                match.group(0)
            )

        except:

            return {}


    if not isinstance(
        data,
        dict
    ):

        return {}


    return data


# ============================================================
# DETERMINISTIC FALLBACKS
#
# This specifically protects against cases where the LLM
# misses obvious values in the text.
# ============================================================

def deterministic_extract(
    text: str,
    field: str,
    field_type: str
):

    field_normalized = normalize_field_name(
        field
    )

    lower = text.lower()


    # ========================================================
    # BANK RELATIONSHIPS
    # ========================================================

    if (
        "bank" in field_normalized
        and field_normalized.startswith(
            "from_"
        )
    ):

        # "from HDFC to SBI"
        pattern = re.search(
            r"\bfrom\s+([A-Za-z][A-Za-z0-9 .&_-]*?)"
            r"\s+\bto\b",
            text,
            re.IGNORECASE
        )


        if pattern:

            candidate = pattern.group(
                1
            ).strip()


            # Avoid capturing excessive text
            candidate = re.split(
                r"\s+(?:for|on|at|with|using)\s+",
                candidate,
                flags=re.IGNORECASE
            )[0].strip()


            if candidate:

                return convert_value(
                    candidate,
                    field_type
                )


    if (
        "bank" in field_normalized
        and field_normalized.startswith(
            "to_"
        )
    ):

        pattern = re.search(
            r"\bto\s+([A-Za-z][A-Za-z0-9 .&_-]*?)"
            r"(?:\s+(?:for|on|at|with|using)\b|[.,]|$)",
            text,
            re.IGNORECASE
        )


        if pattern:

            candidate = pattern.group(
                1
            ).strip()


            if candidate:

                return convert_value(
                    candidate,
                    field_type
                )


    # ========================================================
    # COMMON NAME FIELDS
    # ========================================================

    if field_normalized in {
        "customer_name",
        "client_name",
        "buyer_name",
        "purchaser_name"
    }:

        patterns = [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:bought|purchased|ordered)",
            r"(?:customer|client|buyer|purchaser)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]+)"
        ]


        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:

                return convert_value(
                    match.group(1).strip(),
                    field_type
                )


    # ========================================================
    # QUANTITY
    # ========================================================

    if field_normalized in {
        "quantity",
        "qty",
        "count",
        "number"
    }:

        match = re.search(
            r"\b(\d+)\b\s+"
            r"(?:items?|units?|pieces?|notebooks?|books?|"
            r"products?|tickets?|copies?)",
            text,
            re.IGNORECASE
        )


        if match:

            return int(
                match.group(1)
            )


    # ========================================================
    # PURCHASE DATE
    # ========================================================

    if field_normalized in {
        "purchase_date",
        "date",
        "transaction_date",
        "order_date"
    }:

        patterns = [

            r"\b\d{1,2}\s+"
            r"(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)"
            r"\s+\d{4}\b",

            r"\b\d{4}-\d{2}-\d{2}\b",

            r"\b\d{1,2}/\d{1,2}/\d{4}\b"

        ]


        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )


            if match:

                return convert_value(
                    match.group(0),
                    "date"
                )


    # ========================================================
    # STORE
    # ========================================================

    if field_normalized in {
        "store",
        "shop",
        "seller",
        "merchant"
    }:

        patterns = [

            r"\bfrom\s+([A-Z][A-Za-z0-9 .&'-]+?)(?:\.|$)",

            r"\b(?:store|shop|merchant|seller)\s*[:\-]\s*"
            r"([A-Za-z0-9 .&'-]+)"

        ]


        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )


            if match:

                candidate = match.group(
                    1
                ).strip()


                if candidate:

                    return candidate


    return None


# ============================================================
# MAIN EXTRACTION
# ============================================================

def dynamic_extract(
    text: str,
    schema: Dict[str, str]
):

    llm_data = extract_with_llm(
        text,
        schema
    )


    result = {}


    for field, field_type in schema.items():

        # ----------------------------------------------------
        # First: use LLM result
        # ----------------------------------------------------

        value = llm_data.get(
            field
        )


        converted = convert_value(
            value,
            field_type
        )


        # ----------------------------------------------------
        # Second: deterministic fallback
        #
        # If LLM returned null, try direct extraction.
        # ----------------------------------------------------

        if converted is None:

            fallback = deterministic_extract(
                text,
                field,
                field_type
            )


            if fallback is not None:

                converted = fallback


        # ----------------------------------------------------
        # Always include field
        # ----------------------------------------------------

        result[field] = converted


    return result


# ============================================================
# ENDPOINT
# ============================================================

@app.post(
    "/dynamic-extract"
)
async def dynamic_extract_endpoint(
    request: DynamicExtractRequest
):

    text = request.text.strip()


    if not text:

        raise HTTPException(
            status_code=400,
            detail="text must not be empty"
        )


    try:

        validate_schema(
            request.schema
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


    try:

        result = dynamic_extract(
            text,
            request.schema
        )


    except Exception as e:

        print(
            "Extraction error:",
            repr(e)
        )


        # Even on extraction failure,
        # return exact requested schema.

        result = {
            field: None
            for field in request.schema
        }


    # --------------------------------------------------------
    # FINAL STRICT SCHEMA ENFORCEMENT
    # --------------------------------------------------------

    final_result = {}


    for field, field_type in request.schema.items():

        value = result.get(
            field
        )


        final_result[field] = convert_value(
            value,
            field_type
        )


    return final_result


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Dynamic Schema Structured Extraction API",
        "endpoint": "/dynamic-extract"
    }