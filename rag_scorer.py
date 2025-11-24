import requests
import json
import time
import re
from typing import Dict, Optional, Tuple

class RAGScorer:
    """Score pairs using RAG with Mistral API"""
    
    def __init__(self, api_key: str, model: str = "open-mistral-7b"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.max_retries = 3
        self.retry_delay = 1
    
    def _create_scoring_prompt(self, company_a, company_b, mgmt_a, mgmt_b, macro_report):
        """Create prompt for scoring pair"""
        
        # Format financial metrics
        financials_a = f"ret_12_1={company_a.get('ret_12_1', 'N/A')}, " \
                      f"log_market_cap={company_a.get('log_market_cap', 'N/A')}, " \
                      f"niq_su={company_a.get('niq_su', 'N/A')}, " \
                      f"ivol_ff3_21d={company_a.get('ivol_ff3_21d', 'N/A')}, " \
                      f"niq_at={company_a.get('niq_at', 'N/A')}"
        
        financials_b = f"ret_12_1={company_b.get('ret_12_1', 'N/A')}, " \
                      f"log_market_cap={company_b.get('log_market_cap', 'N/A')}, " \
                      f"niq_su={company_b.get('niq_su', 'N/A')}, " \
                      f"ivol_ff3_21d={company_b.get('ivol_ff3_21d', 'N/A')}, " \
                      f"niq_at={company_b.get('niq_at', 'N/A')}"
        
        # Handle missing mgmt text (now using summarized versions)
        mgmt_text_a = mgmt_a if mgmt_a else "No management discussion text available."
        mgmt_text_b = mgmt_b if mgmt_b else "No management discussion text available."
        
        # Summaries are already compressed, but truncate if still too long for API limits (keep first 6000 chars)
        if len(mgmt_text_a) > 6000:
            mgmt_text_a = mgmt_text_a[:6000] + "... [truncated]"
        if len(mgmt_text_b) > 6000:
            mgmt_text_b = mgmt_text_b[:6000] + "... [truncated]"
        
        prompt = f"""You are a financial analyst comparing two stocks to determine if their portfolio weights should be swapped.

Company A (Current Positive Weight: {company_a.get('weight_change', 0)}):
- ID: {company_a.get('id', 'N/A')}
- Financial Metrics: {financials_a}
- Management Discussion Text:
{mgmt_text_a}

Company B (Current Negative Weight: {company_b.get('weight_change', 0)}):
- ID: {company_b.get('id', 'N/A')}
- Financial Metrics: {financials_b}
- Management Discussion Text:
{mgmt_text_b}

Macro Environment Context:
{macro_report}

Based PRIMARY on the management discussion texts and macro context, score each company (1-10 scale) on these 5 dimensions:

1. Long-term Growth: Prospects for sustained growth, expansion plans, market positioning
2. Cash Generation: Quality of operating cash flow, capital efficiency, working capital management
3. Future Earning Yields: Earnings quality, margin sustainability, revenue visibility
4. Downside Risks: Risk factors, debt levels, competitive threats, regulatory risks
5. Macro Condition Sensitivity: How current macro conditions affect each company differently

Provide your response in the following JSON format:
{{
    "company_a_scores": {{
        "long_term_growth": <score 1-10>,
        "cash_generation": <score 1-10>,
        "future_earning_yields": <score 1-10>,
        "downside_risks": <score 1-10, higher = more risky>,
        "macro_sensitivity": <score 1-10>
    }},
    "company_b_scores": {{
        "long_term_growth": <score 1-10>,
        "cash_generation": <score 1-10>,
        "future_earning_yields": <score 1-10>,
        "downside_risks": <score 1-10, higher = more risky>,
        "macro_sensitivity": <score 1-10>
    }},
    "swap_decision": <0 or 1, where 1 means swap weights>,
    "urgency": "<required> or <optional>",
    "justification": "<brief explanation of decision based on score comparison>"
}}

Important: Base your scores primarily on the textual analysis (management discussions and macro context), not just financial metrics. 

The swap decision should be based on comparing the 5 scores. Only recommend swap=1 if you are VERY CONFIDENT that Company B is significantly better. This means:
- Company B must score better on at least 4 out of 5 dimensions, AND
- The score differences should be meaningful (not just 1 point), AND
- Company B should have notably better scores on critical dimensions (long_term_growth, cash_generation, future_earning_yields).
Only set swap_decision=1 if you have high confidence. When in doubt, set swap_decision=0."""
        
        return prompt
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text that may contain other content"""
        if not text:
            return None
        
        # Remove markdown code blocks
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        # Find first brace
        first_brace = text.find('{')
        if first_brace == -1:
            return None
        
        text = text[first_brace:]
        
        # Use brace counting to find complete JSON object
        brace_count = 0
        i = 0
        while i < len(text):
            char = text[i]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[:i+1]
            i += 1
        
        # Fallback: regex match
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
        
        return None
    
    def _fix_json_common_issues(self, json_str: str) -> str:
        """Fix common JSON syntax errors"""
        # Remove comments
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Remove trailing commas before } or ]
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix single quotes in property names (only if not already double-quoted)
        json_str = re.sub(r"'(\w+)'(\s*):", r'"\1"\2:', json_str)
        
        # Fix single quotes in string values (be careful not to break already quoted strings)
        # Only replace single quotes that are clearly string delimiters
        json_str = re.sub(r':\s*\'([^\']*)\'(\s*[,}])', r': "\1"\2', json_str)
        
        # Fix escaped single quotes
        json_str = json_str.replace("\\'", "'")
        
        return json_str
    
    def _parse_json_safely(self, content: str) -> Optional[Dict]:
        """Parse JSON with multiple fallback strategies"""
        if not content or not content.strip():
            return None
        
        content = content.strip()
        
        # Strategy 1: Try direct parsing
        try:
            decoder = json.JSONDecoder()
            obj, idx = decoder.raw_decode(content)
            return obj
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 2: Fix common issues and try again
        fixed = self._fix_json_common_issues(content)
        try:
            decoder = json.JSONDecoder()
            obj, idx = decoder.raw_decode(fixed)
            return obj
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 3: Extract JSON from text
        extracted = self._extract_json_from_text(content)
        if extracted:
            try:
                decoder = json.JSONDecoder()
                obj, idx = decoder.raw_decode(extracted)
                return obj
            except (json.JSONDecodeError, ValueError):
                # Try fixing extracted JSON
                fixed_extracted = self._fix_json_common_issues(extracted)
                try:
                    decoder = json.JSONDecoder()
                    obj, idx = decoder.raw_decode(fixed_extracted)
                    return obj
                except (json.JSONDecodeError, ValueError):
                    pass
        
        return None
    
    def _validate_and_fix_json(self, parsed: Dict) -> Optional[Dict]:
        """Validate JSON structure and fill missing values with defaults"""
        required_keys = ['company_a_scores', 'company_b_scores', 'swap_decision', 'urgency', 'justification']
        
        # Check all required keys exist
        for key in required_keys:
            if key not in parsed:
                return None
        
        # Validate scores structure
        score_keys = ['long_term_growth', 'cash_generation', 'future_earning_yields', 'downside_risks', 'macro_sensitivity']
        
        for company_key in ['company_a_scores', 'company_b_scores']:
            if company_key not in parsed:
                return None
            scores = parsed[company_key]
            for score_key in score_keys:
                if score_key not in scores:
                    scores[score_key] = 5  # Default
                else:
                    try:
                        scores[score_key] = float(scores[score_key])
                        scores[score_key] = max(1, min(10, scores[score_key]))  # Clamp to 1-10
                    except (ValueError, TypeError):
                        scores[score_key] = 5
        
        # Validate swap_decision
        try:
            swap = int(parsed['swap_decision'])
            parsed['swap_decision'] = 1 if swap >= 1 else 0
        except (ValueError, TypeError):
            parsed['swap_decision'] = 0
        
        # Validate urgency
        if parsed['urgency'] not in ['required', 'optional']:
            parsed['urgency'] = 'optional'
        
        # Ensure justification is a string
        if not isinstance(parsed['justification'], str):
            parsed['justification'] = str(parsed.get('justification', ''))
        
        return parsed
    
    def _check_swap_confidence(self, parsed: Dict) -> Dict:
        """Override swap_decision to be more conservative - only swap when very confident"""
        scores_a = parsed.get('company_a_scores', {})
        scores_b = parsed.get('company_b_scores', {})
        
        score_keys = ['long_term_growth', 'cash_generation', 'future_earning_yields', 'downside_risks', 'macro_sensitivity']
        
        # Count how many dimensions Company B is better
        b_better_count = 0
        significant_wins = 0  # Wins by 2+ points
        critical_wins = 0  # Wins on critical dimensions
        
        critical_dimensions = ['long_term_growth', 'cash_generation', 'future_earning_yields']
        
        for key in score_keys:
            score_a = scores_a.get(key, 5)
            score_b = scores_b.get(key, 5)
            
            # For downside_risks, lower is better (so B is better if score_b < score_a)
            if key == 'downside_risks':
                if score_b < score_a:
                    b_better_count += 1
                    if (score_a - score_b) >= 2:
                        significant_wins += 1
                    if key in critical_dimensions:
                        critical_wins += 1
            else:
                # For other dimensions, higher is better
                if score_b > score_a:
                    b_better_count += 1
                    if (score_b - score_a) >= 2:
                        significant_wins += 1
                    if key in critical_dimensions:
                        critical_wins += 1
        
        # Conservative criteria: Only swap if:
        # 1. B is better on at least 4 out of 5 dimensions, AND
        # 2. At least 2 significant wins (2+ point difference), AND
        # 3. B wins on at least 2 critical dimensions
        should_swap = (
            b_better_count >= 4 and
            significant_wins >= 2 and
            critical_wins >= 2
        )
        
        # Override swap_decision if not confident enough
        if parsed.get('swap_decision', 0) == 1 and not should_swap:
            parsed['swap_decision'] = 0
            parsed['urgency'] = 'optional'  # Also downgrade urgency
            if 'justification' in parsed:
                parsed['justification'] = f"[Conservative override] {parsed.get('justification', '')}"
        
        return parsed
    
    def _call_api(self, prompt: str) -> Optional[Dict]:
        """Call Mistral API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON-only API. You must respond with ONLY valid JSON. No explanations, no text before or after. Your response must be a valid JSON object that can be parsed directly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Very low for consistent JSON
            "max_tokens": 800   # Reduced to avoid truncation
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    
                    # Check if it's a refusal/safety response
                    refusal_keywords = [
                        "I'm an AI",
                        "I don't have the ability",
                        "I cannot",
                        "I'm not able",
                        "I cannot directly"
                    ]
                    if any(keyword.lower() in content.lower() for keyword in refusal_keywords):
                        print(f"Warning: Model refused request. Response: {content[:150]}")
                        if attempt < self.max_retries - 1:
                            # Try with a more direct prompt on retry
                            payload["messages"][1]["content"] = prompt + "\n\nCRITICAL: Your response must START with { and END with }. Return ONLY the JSON object, nothing else."
                            continue
                        return None
                    
                    # Parse with safe method
                    parsed = self._parse_json_safely(content)
                    
                    if parsed:
                        # Validate and fix the parsed JSON
                        validated = self._validate_and_fix_json(parsed)
                        if validated:
                            # Apply conservative confidence check
                            validated = self._check_swap_confidence(validated)
                            return validated
                        else:
                            print(f"Warning: JSON structure invalid. Keys: {list(parsed.keys()) if parsed else 'None'}")
                            if attempt < self.max_retries - 1:
                                continue
                    else:
                        # Check if response has any JSON-like content
                        if '{' not in content:
                            print(f"Warning: No JSON found in response. Preview: {content[:200]}")
                            if attempt < self.max_retries - 1:
                                payload["messages"][1]["content"] = prompt + "\n\nCRITICAL: Return ONLY JSON starting with { and ending with }."
                                continue
                        else:
                            if attempt == 0:
                                print(f"Warning: Found {{ but could not parse. Preview: {content[:300]}")
                            if attempt < self.max_retries - 1:
                                continue
                    return None
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"API error (status {response.status_code}): {response.text[:200]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    return None
                    
            except json.JSONDecodeError as e:
                # This is the "Extra data" or "Expecting value" error
                # Try to extract just the first JSON object
                if 'response' in locals():
                    try:
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                        # Use raw_decode to get only first JSON object
                        decoder = json.JSONDecoder()
                        obj, idx = decoder.raw_decode(content)
                        validated = self._validate_and_fix_json(obj)
                        if validated:
                            return validated
                    except:
                        pass
                
                print(f"JSON decode error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                return None
            except Exception as e:
                print(f"Error calling API (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                return None
        
        return None
    
    def score_pair(self, company_a, company_b, mgmt_a, mgmt_b, macro_report) -> Optional[Dict]:
        """
        Score a pair of companies
        
        Returns:
            Dict with scores, swap_decision, urgency, justification or None if failed
        """
        prompt = self._create_scoring_prompt(company_a, company_b, mgmt_a, mgmt_b, macro_report)
        result = self._call_api(prompt)
        return result

