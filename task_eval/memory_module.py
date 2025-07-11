"""
Memory module for LoCoMo benchmark evaluation.

This module provides a baseline memory system that can be integrated with any LLM
to enhance conversational memory capabilities during evaluation.

Based on: https://gist.github.com/codelion/6cbbd3ec7b0ccef77d3c1fe3d6b0a57c
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
import uuid
import json

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    id: str
    memory: str  
    embedding: np.ndarray
    user_id: str

class BaselineMemory:
    def __init__(self, model_client: Any, model_name: str, debug: bool = False):
        """Initialize baseline memory system.
        
        Args:
            model_client: The LLM client (e.g., OpenAI, Anthropic, Gemini client)
            model_name: The model name being used for evaluation
            debug: Enable debug output for memory retrieval
        """
        self.client = model_client
        self.model_name = model_name
        self.debug = debug
        # Initialize sentence transformer model for embeddings
        # Using all-mpnet-base-v2 for better semantic understanding
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        # Storage for memories, keyed by user_id
        self.memories: Dict[str, List[Memory]] = {}

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(texts)

    def _parse_json_response(self, response_text: str) -> List[str]:
        """Robustly parse JSON array from LLM response, handling various formats."""
        try:
            # Clean the response text
            text = response_text.strip()
            
            # Remove markdown code blocks if present
            if '```json' in text:
                # Extract content between ```json and ```
                start_marker = '```json'
                end_marker = '```'
                start_idx = text.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    # Find the next ``` after the start marker
                    end_idx = text.find(end_marker, start_idx)
                    if end_idx != -1:
                        text = text[start_idx:end_idx].strip()
            elif '```' in text:
                # Handle generic code blocks - extract everything between first and last ```
                first_backtick = text.find('```')
                last_backtick = text.rfind('```')
                if first_backtick != -1 and last_backtick != -1 and first_backtick != last_backtick:
                    # Skip the first ``` and any language identifier on the same line
                    start_idx = text.find('\n', first_backtick)
                    if start_idx == -1:
                        start_idx = first_backtick + 3
                    else:
                        start_idx += 1
                    text = text[start_idx:last_backtick].strip()
            
            # Try to find JSON array boundaries
            start = text.find('[')
            end = text.rfind(']') + 1
            
            if start == -1 or end == 0:
                # Try alternative patterns
                # Look for array-like content without brackets
                lines = text.split('\n')
                array_items = []
                for line in lines:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('//') or line.startswith('#'):
                        continue
                    # Look for quoted strings that could be array items
                    if line.startswith('"') and line.endswith('",'):
                        array_items.append(line[:-1])  # Remove trailing comma
                    elif line.startswith('"') and line.endswith('"'):
                        array_items.append(line)
                
                if array_items:
                    # Reconstruct as JSON array
                    json_str = '[' + ','.join(array_items) + ']'
                    return json.loads(json_str)
                
                logger.error(f"Could not find JSON array in response: {response_text[:200]}...")
                return []
            
            # Extract the JSON array portion
            json_str = text[start:end]
            
            # Try to parse directly
            try:
                facts = json.loads(json_str)
                if isinstance(facts, list):
                    return facts
                else:
                    logger.error(f"Parsed JSON is not a list: {type(facts)}")
                    return []
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                json_str = self._fix_json_formatting(json_str)
                try:
                    facts = json.loads(json_str)
                    if isinstance(facts, list):
                        return facts
                    else:
                        logger.error(f"Parsed JSON is not a list: {type(facts)}")
                        return []
                except json.JSONDecodeError:
                    # Last resort: try to extract individual quoted strings
                    logger.warning(f"Failed to parse JSON, attempting to extract individual strings...")
                    extracted_facts = self._extract_strings_from_text(json_str)
                    if extracted_facts:
                        logger.info(f"Extracted {len(extracted_facts)} facts using fallback method")
                        return extracted_facts
                    logger.error(f"Failed to parse JSON even after all fixes: {json_str[:200]}...")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in _parse_json_response: {str(e)}")
            return []

    def _fix_json_formatting(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove trailing commas before closing brackets
        import re
        
        # Remove trailing comma before ]
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Remove trailing comma before }
        json_str = re.sub(r',\s*}', '}', json_str)
        
        # Handle single quotes (convert to double quotes for valid JSON)
        # This is more complex as we need to avoid converting quotes inside strings
        lines = json_str.split('\n')
        fixed_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("'") and line.endswith("',"):
                # Convert single quotes to double quotes for array items
                line = '"' + line[1:-2] + '",'
            elif line.startswith("'") and line.endswith("'"):
                # Convert single quotes to double quotes for final array item
                line = '"' + line[1:-1] + '"'
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _extract_strings_from_text(self, text: str) -> List[str]:
        """Extract quoted strings from text as a fallback parsing method."""
        import re
        
        # Find all quoted strings (double quotes)
        pattern = r'"([^"]*)"'
        matches = re.findall(pattern, text)
        
        # Filter out very short or empty strings
        facts = [match.strip() for match in matches if len(match.strip()) > 10]
        
        return facts

    def _extract_facts(self, messages) -> List[str]:
        """Use LLM to extract key facts from conversation messages."""
        # Convert messages to string format
        if isinstance(messages, str):
            conversation = messages
        else:
            conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        prompt = f'''Extract the most important and specific facts from this conversation that would be useful for answering questions later. Focus on concrete, verifiable information.

EXTRACT facts about:
- Names, locations, jobs, and specific details about people
- Important events with dates/times when mentioned
- Relationships between people
- Specific preferences, interests, or behaviors
- Plans, goals, or intentions
- Unique experiences or stories

REQUIREMENTS for each memory:
- Must be a complete, specific sentence
- Must clearly identify WHO the fact is about
- Must be factual and verifiable from the conversation
- Should be useful for answering future questions

AVOID extracting:
- Vague or generic statements
- Temporary emotions or states
- Small talk or filler conversation
- Information that's too obvious or common
- Incomplete thoughts or unclear references

Conversation:
{conversation}

Format each memory as a complete, specific sentence. Return ONLY a JSON array of strings.

Example good memories:
[
    "Sarah Johnson lives in Boston's Back Bay neighborhood",
    "Sarah has been a software engineer at Google since March 2023",
    "Sarah's cat Luna is allergic to fish",
    "Sarah met her best friend Kim during their freshman year at MIT",
    "Sarah plans to run the Chicago Marathon in October 2024",
    "Sarah prefers working early mornings, usually starting at 6 AM"
]

Example memories to avoid:
[
    "Sarah is having a good day",  // temporary state
    "Sarah likes food",  // too generic
    "The weather is nice",  // not a meaningful memory
    "Someone mentioned traveling"  // lacks specific attribution
]

Return ONLY the JSON array of extracted memories, nothing else.'''

        try:
            # Use the appropriate model based on what's being evaluated
            if 'gpt' in self.model_name.lower():
                # OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                    temperature=0
                )
                response_text = response.choices[0].message.content
            elif 'claude' in self.model_name.lower():
                # Anthropic API
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                )
                response_text = response.content[0].text
            elif 'gemini' in self.model_name.lower():
                # Gemini API
                import time
                from global_methods import run_gemini
                # Add a longer delay before memory API calls to reduce connection pressure
                time.sleep(3)
                response_text = run_gemini(self.client, self.model_name, prompt)
                if response_text is None:
                    logger.error("Gemini API returned None for memory extraction")
                    return []
            else:
                logger.error(f"Unsupported model for memory extraction: {self.model_name}")
                return []

            # Extract JSON array from response with robust parsing
            facts = self._parse_json_response(response_text)
            return facts
        except Exception as e:
            logger.error(f"Error extracting facts: {str(e)}")
            return []

    def add_conversation(self, conversation_data: Dict, user_id: str) -> None:
        """Add conversation data to memory by extracting key facts."""
        # Convert conversation sessions to messages format
        messages = []
        for session_key in sorted(conversation_data.keys()):
            if session_key.startswith('session_') and not session_key.endswith('_date_time'):
                session_data = conversation_data[session_key]
                date_key = f"{session_key}_date_time"
                date_info = conversation_data.get(date_key, "")
                
                # Add date context
                if date_info:
                    messages.append(f"[Date: {date_info}]")
                
                # Add session conversations
                for dialog in session_data:
                    speaker = dialog.get('speaker', 'Unknown')
                    text = dialog.get('text', '')
                    
                    # Add image context if present
                    if 'blip_caption' in dialog:
                        text += f" [shared image: {dialog['blip_caption']}]"
                    
                    messages.append(f"{speaker}: {text}")
        
        # Extract facts from the full conversation
        conversation_text = "\n".join(messages)
        self.add(conversation_text, user_id)

    def add(self, messages, user_id: str) -> None:
        """Add new memories from conversation messages."""
        # Extract facts using LLM
        facts = self._extract_facts(messages)
        
        # Generate embeddings for facts
        if facts:
            embeddings = self._get_embeddings(facts)
            
            # Create Memory objects
            new_memories = [
                Memory(
                    id=str(uuid.uuid4()),
                    memory=fact,
                    embedding=embedding,
                    user_id=user_id
                )
                for fact, embedding in zip(facts, embeddings)
            ]
            
            # Add to storage
            if user_id not in self.memories:
                self.memories[user_id] = []
            self.memories[user_id].extend(new_memories)
            
            logger.info(f"Added {len(new_memories)} memories for user {user_id}")

    def search(self, query: str, user_id: str, limit: int = 32) -> List[Dict[str, str]]:
        """Search memories using embedding similarity."""
        if user_id not in self.memories:
            return []
            
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Get user's memories
        user_memories = self.memories[user_id]
        if not user_memories:
            return []
            
        # Calculate similarities
        similarities = [np.dot(query_embedding, mem.embedding) / 
                       (np.linalg.norm(query_embedding) * np.linalg.norm(mem.embedding))
                       for mem in user_memories]
                       
        # Sort by similarity
        memory_similarities = list(zip(user_memories, similarities))
        memory_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches
        results = []
        for memory, similarity in memory_similarities[:limit]:
            if similarity > 0.1:  # Similarity threshold
                results.append({
                    "id": memory.id,
                    "memory": memory.memory,
                    "similarity": similarity
                })
                
        return results

    def get_relevant_context(self, question: str, user_id: str, max_memories: int = 5) -> str:
        """Get relevant memory context for a question."""
        relevant_memories = self.search(question, user_id, limit=max_memories)
        
        if not relevant_memories:
            return ""
        
        # Filter for higher quality memories (tuned for all-mpnet-base-v2)
        # MPNet typically produces higher similarity scores, so we can use a more moderate threshold
        high_quality_memories = [mem for mem in relevant_memories if mem['similarity'] > 0.4]
        
        # If no high-quality memories, return empty (better than confusing the model)
        if not high_quality_memories:
            return ""
        
        # Limit to top 1-2 most relevant to avoid overloading
        top_memories = high_quality_memories[:2]
        
        # Debug output
        if self.debug:
            print(f"Debug: Retrieved {len(top_memories)} relevant memories for question: {question[:50]}...")
            for i, mem in enumerate(top_memories):
                print(f"  Memory {i+1} (sim: {mem['similarity']:.3f}): {mem['memory'][:80]}...")
        
        # Format memories as additional context without disrupting the flow
        context_lines = []
        for mem in top_memories:
            # Add memories as natural context sentences
            context_lines.append(mem['memory'])
        
        # Return as a paragraph of context
        if context_lines:
            return "Additional context: " + " ".join(context_lines) + "\n\n"
        else:
            return ""

    def update(self, memory_id: str, new_content: str) -> None:
        """Update a specific memory with new content."""
        # Find memory
        for memories in self.memories.values():
            for i, memory in enumerate(memories):
                if memory.id == memory_id:
                    # Create new embedding for updated content
                    new_embedding = self._get_embeddings([new_content])[0]
                    # Update memory
                    memories[i] = Memory(
                        id=memory.id,
                        memory=new_content,
                        embedding=new_embedding,
                        user_id=memory.user_id
                    )
                    return
                    
        logger.warning(f"Memory {memory_id} not found")

    def delete_all(self, user_id: str) -> None:
        """Delete all memories for a specific user.
        
        Args:
            user_id: ID of user whose memories should be deleted
        """
        if user_id in self.memories:
            del self.memories[user_id]
            
    def get_memory_stats(self, user_id: str) -> Dict[str, int]:
        """Get statistics about stored memories for a user."""
        if user_id not in self.memories:
            return {"total_memories": 0}
        
        return {
            "total_memories": len(self.memories[user_id])
        }