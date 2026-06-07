"""Advanced context analysis module for AI Agent"""
import re
from typing import Dict, List


class ContextAnalyzer:
    def __init__(self):
        self.intent_patterns = {
            "code_generation": [
                r"write\s+(a\s+)?(python|javascript|java|c\+\+|ruby|go|rust|swift)",
                r"create\s+(a\s+)?(function|class|module|script)",
                r"implement\s+(a\s+)?",
                r"build\s+(a\s+)?(tool|utility|helper)",
                r"generate\s+(a\s+)?(code|script|program)",
            ],
            "code_analysis": [
                r"analyze\s+(the\s+)?(code|performance|complexity)",
                r"review\s+(this\s+)?(code|pull request)",
                r"check\s+(for\s+)?(bugs|errors|issues)",
                r"optimize\s+(this\s+)?",
                r"refactor\s+(this\s+)?",
            ],
            "web_search": [
                r"search\s+(for\s+)?",
                r"find\s+(me\s+)?",
                r"look\s+up",
                r"google\s+",
                r"what\s+(is|are|was|were)\s+",
                r"latest\s+(news|trends|updates)",
            ],
            "file_operation": [
                r"read\s+(the\s+)?file",
                r"write\s+(to\s+)?file",
                r"edit\s+(the\s+)?file",
                r"delete\s+(the\s+)?file",
                r"create\s+(a\s+)?file",
            ],
            "data_analysis": [
                r"analyze\s+(the\s+)?data",
                r"show\s+(me\s+)?(statistics|stats|metrics)",
                r"create\s+(a\s+)?(chart|graph|visualization)",
                r"calculate\s+(the\s+)?",
                r"summarize\s+(this\s+)?",
            ],
            "explanation": [
                r"explain\s+(this|how|what|why)",
                r"what\s+(does|is|are)\s+",
                r"how\s+(does|do|can|should)\s+",
                r"tell\s+me\s+about",
                r"describe\s+",
            ],
        }

        self.entity_patterns = {
            "file_path": [
                r'[\w/\\.-]+\.(?:py|js|ts|java|cpp|rs|go|rb|php|html|css|json|xml|csv|txt|md|c|cs)',
                r'(?:C:|D:|\/home|\/usr|\/etc)\\?[\w\/.-]+',
            ],
            "programming_language": [
                r'\b(python|javascript|java|typescript|c\+\+|c#|ruby|go|rust|swift|kotlin|php|scala|r|matlab|sql)\b',
            ],
            "framework": [
                r'\b(django|flask|fastapi|react|vue|angular|node\.js|express|spring\.net|rails|laravel)\b',
            ],
            "concept": [
                r'\b(algorithm|data structure|design pattern|api|database|cache|queue|stack|tree|graph)\b',
            ],
        }

    def analyze_intent(self, text: str) -> Dict:
        """Analyze the intent of user input"""
        text_lower = text.lower()
        scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[intent] = score

        if max(scores.values()) == 0:
            return {"intent": "general", "confidence": 0.5}

        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 3, 1.0)

        return {
            "intent": best_intent,
            "confidence": confidence,
            "all_scores": scores
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {}

        for entity_type, patterns in self.entity_patterns.items():
            found = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found.extend(matches)
            if found:
                entities[entity_type] = list(set(found))

        return entities

    def calculate_relevance_score(self, query: str, context: str) -> float:
        """Calculate relevance score between query and context"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))

        if not query_words:
            return 0.0

        intersection = query_words & context_words
        score = len(intersection) / len(query_words)

        return min(score * 1.5, 1.0)

    def get_suggested_tools(self, intent: str, entities: Dict) -> List[str]:
        """Suggest tools based on intent and entities"""
        tool_suggestions = {
            "code_generation": ["run_code", "write_file"],
            "code_analysis": ["code_review", "complexity_metrics", "import_analysis"],
            "web_search": ["search_web", "fetch_url"],
            "file_operation": ["read_file", "write_file", "edit_file", "list_directory"],
            "data_analysis": ["analyze_file", "get_statistics", "create_chart"],
            "explanation": ["code_review"],
        }

        tools = tool_suggestions.get(intent, [])

        if "programming_language" in entities:
            tools.append("run_code")

        if "file_path" in entities:
            tools.extend(["read_file", "file_info"])

        return list(set(tools))
