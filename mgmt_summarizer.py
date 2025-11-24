import requests
import json
import time
import hashlib
import os
import pickle
from typing import Optional, Dict, List

class MgmtSummarizer:
    """Summarize management discussion text using RAG, preserving predictive semantics"""
    
    def __init__(self, api_key: str, model: str = "open-mistral-7b", cache_dir: str = "mgmt_summaries_cache"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.cache_dir = cache_dir
        self.max_retries = 3
        self.retry_delay = 1
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache for recently summarized texts
        self.memory_cache = {}
    
    def _get_cache_key(self, mgmt_text: str) -> str:
        """Generate cache key from text hash"""
        return hashlib.md5(mgmt_text.encode('utf-8')).hexdigest()
    
    def _load_cached_summary(self, cache_key: str) -> Optional[str]:
        """Load summary from disk cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _save_cached_summary(self, cache_key: str, summary: str):
        """Save summary to disk cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(summary, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract major sections from mgmt text"""
        sections = {
            'overview': '',
            'liquidity': '',
            'operations': '',
            'risks': '',
            'segments': '',
            'other': ''
        }
        
        text_lower = text.lower()
        
        # Keywords for section identification
        section_keywords = {
            'overview': ['overview', 'highlights', 'executive summary', 'summary', 'introduction'],
            'liquidity': ['liquidity', 'capital resources', 'cash flow', 'working capital', 'financing'],
            'operations': ['results of operations', 'operating results', 'revenue', 'earnings', 'income'],
            'risks': ['risk factors', 'risks', 'uncertainties', 'forward-looking', 'cautionary'],
            'segments': ['segment', 'business segment', 'geographic', 'product line']
        }
        
        # Simple section extraction - split by common headers
        lines = text.split('\n')
        current_section = 'other'
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            for section, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords) and len(line_lower) < 100:
                    current_section = section
                    break
            
            sections[current_section] += line + '\n'
        
        return sections
    
    def _split_into_chunks(self, text: str, chunk_size: int = 7000, overlap: int = 500) -> List[str]:
        """Split text into chunks with overlap, trying to break at sentence boundaries"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap (last few sentences of previous chunk)
                overlap_sentences = current_chunk.split('. ')[-3:]  # Last 3 sentences
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
            else:
                current_chunk += sentence + '. '
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_summarization_prompt(self, mgmt_text: str, chunk_number: int = None, total_chunks: int = None) -> str:
        """Create prompt for summarization that preserves predictive semantics"""
        
        chunk_info = ""
        if chunk_number is not None and total_chunks is not None:
            chunk_info = f"\n\n[This is chunk {chunk_number} of {total_chunks}. Summarize this section comprehensively, preserving all predictive elements.]"
        
        prompt = f"""Summarize this management discussion text for financial decision-making. Your summary must preserve predictive and decision-relevant information.

CRITICAL REQUIREMENTS:
1. Preserve forward-looking language: "expect", "forecast", "plan", "anticipate", "outlook", "guidance", "projected"
2. Keep risk qualifiers: "may", "uncertain", "volatility", "liquidity risk", "could", "might", "subject to"
3. CRITICAL - Preserve sentiment/tonality: Maintain the original positive/negative tone. If the original text is cautious, keep it cautious. If optimistic, keep it optimistic. Retain sentiment drivers: positive/negative modifiers ("strong demand", "cost pressure", "margin compression", "growth", "decline", "challenges", "opportunities")
4. Preserve causal phrases: "due to", "driven by", "resulting from", "because of", "as a result"
5. Keep ALL numbers, percentages, and directional cues: "increased by 10%", "declined", "record high", "decreased from X to Y"
6. Retain key named entities: company names, product names, geographic regions, customer names
7. Maintain section structure if identifiable: Keep Overview, Liquidity, Operations, Risks, and Segments sections separate
8. PRESERVE TONALITY: If management expresses concern, keep that concern. If they express confidence, keep that confidence. Do not neutralize or change the emotional tone.

Text to summarize:
{mgmt_text}{chunk_info}

Return the summary preserving all predictive elements. If the text contains section headers, preserve them. Otherwise, summarize the content while maintaining the structure.

Rules:
- Preserve ALL numbers, percentages, and specific metrics
- Keep forward-looking tone and language
- Retain named entities (company names, products, regions)
- Compress repetitive historical facts but keep predictive elements
- Target length: 30-40% of original, but prioritize information density over length

Return ONLY the summary, no explanations."""

        return prompt
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Mistral API for summarization"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial text summarizer. Summarize management discussion text while preserving predictive semantics, forward-looking language, risk qualifiers, numbers, and key entities."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,  # Low temperature for consistent summarization
            "max_tokens": 4000   # Allow longer summaries
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=90
                )
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result['choices'][0]['message']['content'].strip()
                    return summary
                
                elif response.status_code == 429:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"API error (status {response.status_code}): {response.text[:200]}")
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
    
    def summarize(self, mgmt_text: str) -> Optional[str]:
        """
        Summarize mgmt text with predictive semantics preservation
        
        Args:
            mgmt_text: Raw management discussion text
            
        Returns:
            Summarized text or None if summarization fails
        """
        if not mgmt_text or len(mgmt_text.strip()) == 0:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(mgmt_text)
        
        # Check memory cache
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cached = self._load_cached_summary(cache_key)
        if cached:
            self.memory_cache[cache_key] = cached
            return cached
        
        # Skip summarization if text is already short (< 1000 chars)
        if len(mgmt_text) < 1000:
            self.memory_cache[cache_key] = mgmt_text
            self._save_cached_summary(cache_key, mgmt_text)
            return mgmt_text
        
        # Split into chunks if text is too long
        chunks = self._split_into_chunks(mgmt_text, chunk_size=7000, overlap=500)
        
        if len(chunks) == 1:
            # Single chunk - summarize directly
            prompt = self._create_summarization_prompt(chunks[0])
            summary = self._call_api(prompt)
            
            if summary:
                self.memory_cache[cache_key] = summary
                self._save_cached_summary(cache_key, summary)
                return summary
            else:
                print("Warning: Summarization failed, returning original text")
                return mgmt_text
        else:
            # Multiple chunks - summarize each and concatenate
            print(f"  Summarizing {len(chunks)} chunks...")
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                print(f"    Processing chunk {i+1}/{len(chunks)}...", end=' ')
                prompt = self._create_summarization_prompt(chunk, chunk_number=i+1, total_chunks=len(chunks))
                chunk_summary = self._call_api(prompt)
                
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
                    print("✓")
                    # Small delay between chunks to avoid rate limits
                    time.sleep(0.5)
                else:
                    print("✗ (using original chunk)")
                    # If summarization fails, use original chunk
                    chunk_summaries.append(chunk[:1000] + "... [summarization failed]")
            
            # Concatenate all summaries
            final_summary = "\n\n".join(chunk_summaries)
            
            # Cache the result
            self.memory_cache[cache_key] = final_summary
            self._save_cached_summary(cache_key, final_summary)
            return final_summary

