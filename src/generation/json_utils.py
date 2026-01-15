import json
import re
from typing import Any, Dict


class JsonUtils:
    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """
        Robust JSON extraction:
        - If the model returns extra text, we try to find the first JSON object.
        """
        text = text.strip()

        # direct parse 
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to extract JSON object from the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output.")

        candidate = match.group(0)
        return json.loads(candidate)
