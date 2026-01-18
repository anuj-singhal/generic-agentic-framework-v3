"""
Name Matching Tools Module
===========================

Provides tools for matching and canonicalizing entity names.
Designed to handle large lists (10-20k names) efficiently with:
- Fuzzy matching algorithms
- Batch processing to respect token limits
- Multiple matching strategies
- No web search dependency (runs on-premises)
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any, Tuple, Set
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib

from core.tools_base import tool_registry


# =============================================================================
# NAME MATCHING CONFIGURATION
# =============================================================================

# Batch size for processing (to manage token limits)
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_SIZE = 500

# Similarity thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.55

# Common business suffixes to normalize
BUSINESS_SUFFIXES = [
    'LLC', 'L.L.C.', 'INC', 'INC.', 'INCORPORATED', 'CORP', 'CORP.', 'CORPORATION',
    'LTD', 'LTD.', 'LIMITED', 'PLC', 'P.L.C.', 'PJSC', 'P.J.S.C.', 'JSC',
    'CO', 'CO.', 'COMPANY', 'GROUP', 'HOLDINGS', 'HOLDING',
    'INTERNATIONAL', 'INTL', 'INT\'L', 'GLOBAL', 'WORLDWIDE',
    'SERVICES', 'SERVICE', 'SOLUTIONS', 'ENTERPRISES', 'ENTERPRISE',
    'AUTHORITY', 'BANK', 'FINANCIAL', 'INVESTMENT', 'INVESTMENTS',
    'FZ', 'FZE', 'FZC', 'FZCO', 'DWC', 'DMCC',  # UAE specific
    'SA', 'S.A.', 'AG', 'A.G.', 'GMBH', 'BV', 'B.V.', 'NV', 'N.V.',  # European
    'PTY', 'PROPRIETARY', 'PVT', 'PRIVATE',
]

# Common abbreviations mapping
COMMON_ABBREVIATIONS = {
    'INTL': 'INTERNATIONAL',
    'INT\'L': 'INTERNATIONAL',
    'CORP': 'CORPORATION',
    'INC': 'INCORPORATED',
    'LTD': 'LIMITED',
    'CO': 'COMPANY',
    'GOVT': 'GOVERNMENT',
    'GOVT.': 'GOVERNMENT',
    'DEPT': 'DEPARTMENT',
    'DEPT.': 'DEPARTMENT',
    'UNIV': 'UNIVERSITY',
    'UNIV.': 'UNIVERSITY',
    'NATL': 'NATIONAL',
    'NAT\'L': 'NATIONAL',
    'FED': 'FEDERAL',
    'FED.': 'FEDERAL',
    'TECH': 'TECHNOLOGY',
    'TECH.': 'TECHNOLOGY',
    'MGMT': 'MANAGEMENT',
    'SVCS': 'SERVICES',
    'SVC': 'SERVICE',
    'ASSOC': 'ASSOCIATION',
    'ASSN': 'ASSOCIATION',
    'CTR': 'CENTER',
    'CNTR': 'CENTER',
    'GRP': 'GROUP',
    'HQ': 'HEADQUARTERS',
    'MFG': 'MANUFACTURING',
    'DIST': 'DISTRIBUTION',
    'ELEC': 'ELECTRICITY',
    'ELECT': 'ELECTRICITY',
    'AUTH': 'AUTHORITY',
}


# =============================================================================
# IN-MEMORY STORAGE FOR NAME MATCHING SESSIONS
# =============================================================================

# Store for loaded name lists and matching results
_name_store: Dict[str, Dict[str, Any]] = {}
_matching_sessions: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_name(name: str, expand_abbreviations: bool = True) -> str:
    """
    Normalize a name for comparison.
    - Uppercase
    - Remove special characters
    - Optionally expand abbreviations
    - Remove common suffixes
    """
    if not name:
        return ""
    
    # Uppercase
    normalized = name.upper().strip()
    
    # Remove special characters except spaces and alphanumeric
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Expand abbreviations
    if expand_abbreviations:
        words = normalized.split()
        expanded_words = []
        for word in words:
            if word in COMMON_ABBREVIATIONS:
                expanded_words.append(COMMON_ABBREVIATIONS[word])
            else:
                expanded_words.append(word)
        normalized = ' '.join(expanded_words)
    
    return normalized


def remove_suffixes(name: str) -> str:
    """Remove common business suffixes from a name."""
    normalized = normalize_name(name, expand_abbreviations=False)
    words = normalized.split()
    
    # Remove trailing suffixes
    while words and words[-1] in BUSINESS_SUFFIXES:
        words.pop()
    
    # Also check for suffixes anywhere
    filtered_words = [w for w in words if w not in BUSINESS_SUFFIXES]
    
    return ' '.join(filtered_words) if filtered_words else normalized


def get_name_tokens(name: str) -> Set[str]:
    """Get significant tokens from a name."""
    normalized = remove_suffixes(name)
    tokens = set(normalized.split())
    # Remove very short tokens (likely initials or noise)
    tokens = {t for t in tokens if len(t) > 1}
    return tokens


def calculate_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names using multiple methods."""
    # Normalize both names
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match after normalization
    if norm1 == norm2:
        return 1.0
    
    # Remove suffixes and compare
    core1 = remove_suffixes(name1)
    core2 = remove_suffixes(name2)
    
    if core1 == core2:
        return 0.98
    
    # Sequence matching (handles insertions, deletions)
    seq_ratio = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Token-based matching (handles word reordering)
    tokens1 = get_name_tokens(name1)
    tokens2 = get_name_tokens(name2)
    
    if tokens1 and tokens2:
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        jaccard = len(intersection) / len(union) if union else 0
        
        # Check if one is subset of other (e.g., "DEWA" vs "DEWA Authority")
        if tokens1.issubset(tokens2) or tokens2.issubset(tokens1):
            subset_bonus = 0.15
        else:
            subset_bonus = 0
    else:
        jaccard = 0
        subset_bonus = 0
    
    # Weighted combination
    combined_score = (seq_ratio * 0.5) + (jaccard * 0.35) + subset_bonus
    
    # Check if one name contains the other (important for abbreviations)
    if core1 in core2 or core2 in core1:
        combined_score = max(combined_score, 0.80)
    
    return min(combined_score, 1.0)


def generate_session_id(names: List[str]) -> str:
    """Generate a unique session ID based on input names."""
    content = json.dumps(sorted(names[:100]))  # Use first 100 for hash
    return hashlib.md5(content.encode()).hexdigest()[:12]


def create_name_index(names: List[str]) -> Dict[str, List[int]]:
    """Create an inverted index of tokens to name indices for fast lookup."""
    index = defaultdict(list)
    
    for i, name in enumerate(names):
        tokens = get_name_tokens(name)
        for token in tokens:
            index[token].append(i)
        
        # Also index first word and core name
        core = remove_suffixes(name)
        if core:
            first_word = core.split()[0] if core.split() else ""
            if first_word and len(first_word) > 2:
                index[f"_FIRST_{first_word}"].append(i)
    
    return dict(index)


# =============================================================================
# NAME MATCHING TOOLS
# =============================================================================

@tool
def load_names_for_matching(names_json: str, session_name: str = "default") -> str:
    """
    Load a list of names for matching. This prepares the data for efficient processing.
    
    IMPORTANT: For large lists (1000+ names), provide names in batches using this tool
    multiple times with the same session_name, or use 'append' mode.
    
    Args:
        names_json: JSON array of names, e.g., '["Name 1", "Name 2", "Name 3"]'
        session_name: Identifier for this matching session (use same name to append)
    
    Returns:
        Confirmation with statistics about loaded names
    """
    try:
        names = json.loads(names_json)
        if not isinstance(names, list):
            return "Error: Input must be a JSON array of names"
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"
    
    # Clean and deduplicate
    cleaned_names = []
    seen = set()
    for name in names:
        if isinstance(name, str) and name.strip():
            clean = name.strip()
            if clean.lower() not in seen:
                cleaned_names.append(clean)
                seen.add(clean.lower())
    
    # Initialize or append to session
    if session_name in _name_store:
        # Append to existing session
        existing = _name_store[session_name]["names"]
        existing_set = set(n.lower() for n in existing)
        new_names = [n for n in cleaned_names if n.lower() not in existing_set]
        _name_store[session_name]["names"].extend(new_names)
        _name_store[session_name]["index"] = create_name_index(_name_store[session_name]["names"])
        
        return f"""✅ Appended to session '{session_name}':
- New names added: {len(new_names)}
- Total names in session: {len(_name_store[session_name]['names'])}
- Duplicates skipped: {len(cleaned_names) - len(new_names)}

Use 'find_matching_names' or 'batch_match_names' to find matches."""
    else:
        # Create new session
        _name_store[session_name] = {
            "names": cleaned_names,
            "index": create_name_index(cleaned_names),
            "total_original": len(names)
        }
        
        return f"""✅ Created session '{session_name}':
- Names loaded: {len(cleaned_names)}
- Duplicates removed: {len(names) - len(cleaned_names)}
- Index created for fast matching

Use 'find_matching_names' to find matches for a canonical name.
Use 'batch_match_names' to process multiple canonical names."""


tool_registry.register(load_names_for_matching, "name_matching")


@tool
def find_matching_names(
    canonical_name: str,
    session_name: str = "default",
    threshold: float = 0.65,
    max_results: int = 50
) -> str:
    """
    Find all names that match a canonical name from the loaded list.
    
    The matching algorithm:
    1. Normalizes names (uppercase, remove special chars)
    2. Expands common abbreviations
    3. Uses fuzzy matching with multiple algorithms
    4. Scores based on token overlap and sequence similarity
    
    Args:
        canonical_name: The canonical/standard name to match against
        session_name: The session containing the names to search
        threshold: Minimum similarity score (0.0-1.0), default 0.65
        max_results: Maximum number of matches to return
    
    Returns:
        List of matching names with similarity scores
    """
    if session_name not in _name_store:
        available = list(_name_store.keys()) if _name_store else ["none"]
        return f"Session '{session_name}' not found. Available sessions: {', '.join(available)}. Use 'load_names_for_matching' first."
    
    session = _name_store[session_name]
    names = session["names"]
    index = session["index"]
    
    # Get tokens from canonical name for index lookup
    canonical_tokens = get_name_tokens(canonical_name)
    canonical_core = remove_suffixes(canonical_name)
    canonical_first = canonical_core.split()[0] if canonical_core.split() else ""
    
    # Find candidate indices using index (much faster than scanning all)
    candidate_indices = set()
    
    # Add candidates that share tokens
    for token in canonical_tokens:
        if token in index:
            candidate_indices.update(index[token])
    
    # Add candidates with same first word
    if canonical_first and len(canonical_first) > 2:
        first_key = f"_FIRST_{canonical_first}"
        if first_key in index:
            candidate_indices.update(index[first_key])
    
    # If few candidates found, scan more broadly (for very different spellings)
    if len(candidate_indices) < 10:
        # Sample additional names for comparison
        step = max(1, len(names) // 100)
        for i in range(0, len(names), step):
            candidate_indices.add(i)
    
    # Score candidates
    matches = []
    for idx in candidate_indices:
        if idx < len(names):
            name = names[idx]
            score = calculate_similarity(canonical_name, name)
            if score >= threshold:
                matches.append({
                    "name": name,
                    "score": round(score, 4),
                    "confidence": "high" if score >= HIGH_CONFIDENCE_THRESHOLD else 
                                 "medium" if score >= MEDIUM_CONFIDENCE_THRESHOLD else "low"
                })
    
    # Sort by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:max_results]
    
    # Format result
    if not matches:
        return f"""No matches found for '{canonical_name}' with threshold {threshold}.

Try:
- Lowering the threshold (e.g., 0.5)
- Checking if names are loaded: use 'get_session_info'
- Verifying the canonical name spelling"""
    
    result = [
        f"## Matches for: {canonical_name}",
        f"*Found {len(matches)} matches (threshold: {threshold})*",
        "",
        "| Rank | Name | Score | Confidence |",
        "|------|------|-------|------------|"
    ]
    
    for i, match in enumerate(matches, 1):
        result.append(f"| {i} | {match['name']} | {match['score']:.2f} | {match['confidence']} |")
    
    # Group by confidence
    high = [m for m in matches if m['confidence'] == 'high']
    medium = [m for m in matches if m['confidence'] == 'medium']
    low = [m for m in matches if m['confidence'] == 'low']
    
    result.extend([
        "",
        f"**Summary:** {len(high)} high, {len(medium)} medium, {len(low)} low confidence matches"
    ])
    
    return "\n".join(result)


tool_registry.register(find_matching_names, "name_matching")


@tool
def batch_match_names(
    canonical_names_json: str,
    session_name: str = "default",
    threshold: float = 0.65,
    max_matches_per_name: int = 20
) -> str:
    """
    Match multiple canonical names against the loaded name list in batch.
    More efficient than calling find_matching_names repeatedly.
    
    Args:
        canonical_names_json: JSON array of canonical names to match
        session_name: The session containing names to search
        threshold: Minimum similarity score (0.0-1.0)
        max_matches_per_name: Maximum matches to return per canonical name
    
    Returns:
        Batch matching results with all canonical names and their matches
    """
    try:
        canonical_names = json.loads(canonical_names_json)
        if not isinstance(canonical_names, list):
            return "Error: Input must be a JSON array of canonical names"
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"
    
    if session_name not in _name_store:
        return f"Session '{session_name}' not found. Use 'load_names_for_matching' first."
    
    session = _name_store[session_name]
    names = session["names"]
    index = session["index"]
    
    results = []
    total_matches = 0
    
    for canonical_name in canonical_names:
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            continue
        
        canonical_name = canonical_name.strip()
        
        # Get candidates using index
        canonical_tokens = get_name_tokens(canonical_name)
        canonical_core = remove_suffixes(canonical_name)
        canonical_first = canonical_core.split()[0] if canonical_core.split() else ""
        
        candidate_indices = set()
        for token in canonical_tokens:
            if token in index:
                candidate_indices.update(index[token])
        
        if canonical_first and len(canonical_first) > 2:
            first_key = f"_FIRST_{canonical_first}"
            if first_key in index:
                candidate_indices.update(index[first_key])
        
        # Score and collect matches
        matches = []
        for idx in candidate_indices:
            if idx < len(names):
                name = names[idx]
                score = calculate_similarity(canonical_name, name)
                if score >= threshold:
                    matches.append({"name": name, "score": round(score, 4)})
        
        matches.sort(key=lambda x: x["score"], reverse=True)
        matches = matches[:max_matches_per_name]
        
        results.append({
            "canonical_name": canonical_name,
            "match_count": len(matches),
            "matches": matches
        })
        total_matches += len(matches)
    
    # Format output
    output = [
        "## Batch Matching Results",
        f"*Processed {len(results)} canonical names, found {total_matches} total matches*",
        ""
    ]
    
    for r in results:
        output.append(f"### {r['canonical_name']}")
        if r['matches']:
            output.append(f"*{r['match_count']} matches found*")
            for m in r['matches'][:10]:  # Show top 10
                output.append(f"- {m['name']} (score: {m['score']:.2f})")
            if r['match_count'] > 10:
                output.append(f"- ... and {r['match_count'] - 10} more")
        else:
            output.append("*No matches found*")
        output.append("")
    
    return "\n".join(output)


tool_registry.register(batch_match_names, "name_matching")


@tool
def create_canonical_mapping(
    canonical_name: str,
    session_name: str = "default",
    threshold: float = 0.65,
    include_canonical: bool = True
) -> str:
    """
    Create a mapping from a canonical name to all its variations.
    Returns a structured mapping that can be used for data standardization.
    
    Args:
        canonical_name: The standard/canonical name
        session_name: Session with loaded names
        threshold: Minimum similarity threshold
        include_canonical: Whether to include canonical name in variations
    
    Returns:
        JSON mapping of canonical name to variations
    """
    if session_name not in _name_store:
        return f"Session '{session_name}' not found."
    
    session = _name_store[session_name]
    names = session["names"]
    index = session["index"]
    
    # Find matches
    canonical_tokens = get_name_tokens(canonical_name)
    canonical_core = remove_suffixes(canonical_name)
    canonical_first = canonical_core.split()[0] if canonical_core.split() else ""
    
    candidate_indices = set()
    for token in canonical_tokens:
        if token in index:
            candidate_indices.update(index[token])
    
    if canonical_first and len(canonical_first) > 2:
        first_key = f"_FIRST_{canonical_first}"
        if first_key in index:
            candidate_indices.update(index[first_key])
    
    variations = []
    for idx in candidate_indices:
        if idx < len(names):
            name = names[idx]
            score = calculate_similarity(canonical_name, name)
            if score >= threshold:
                variations.append({"name": name, "score": score})
    
    variations.sort(key=lambda x: x["score"], reverse=True)
    
    # Create mapping
    mapping = {
        "canonical_name": canonical_name,
        "variations": [v["name"] for v in variations],
        "variation_count": len(variations),
        "scores": {v["name"]: round(v["score"], 4) for v in variations}
    }
    
    if include_canonical and canonical_name not in mapping["variations"]:
        mapping["variations"].insert(0, canonical_name)
        mapping["variation_count"] += 1
        mapping["scores"][canonical_name] = 1.0
    
    result = [
        "## Canonical Mapping Created",
        "",
        f"**Canonical Name:** {canonical_name}",
        f"**Variations Found:** {mapping['variation_count']}",
        "",
        "### Mapping (JSON):",
        "```json",
        json.dumps(mapping, indent=2),
        "```",
        "",
        "### Variations List:",
    ]
    
    for v in mapping["variations"][:20]:
        score = mapping["scores"].get(v, 0)
        result.append(f"- {v} ({score:.2f})")
    
    if len(mapping["variations"]) > 20:
        result.append(f"- ... and {len(mapping['variations']) - 20} more")
    
    return "\n".join(result)


tool_registry.register(create_canonical_mapping, "name_matching")


@tool
def bulk_create_mappings(
    canonical_names_json: str,
    session_name: str = "default",
    threshold: float = 0.65,
    output_format: str = "summary"
) -> str:
    """
    Create canonical mappings for multiple names at once.
    Efficient for processing many canonical names.
    
    Args:
        canonical_names_json: JSON array of canonical names
        session_name: Session with loaded names
        threshold: Minimum similarity threshold
        output_format: 'summary', 'detailed', or 'json'
    
    Returns:
        Bulk mappings in specified format
    """
    try:
        canonical_names = json.loads(canonical_names_json)
        if not isinstance(canonical_names, list):
            return "Error: Input must be a JSON array"
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"
    
    if session_name not in _name_store:
        return f"Session '{session_name}' not found."
    
    session = _name_store[session_name]
    names = session["names"]
    index = session["index"]
    
    all_mappings = []
    
    for canonical_name in canonical_names:
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            continue
        
        canonical_name = canonical_name.strip()
        canonical_tokens = get_name_tokens(canonical_name)
        canonical_core = remove_suffixes(canonical_name)
        canonical_first = canonical_core.split()[0] if canonical_core.split() else ""
        
        candidate_indices = set()
        for token in canonical_tokens:
            if token in index:
                candidate_indices.update(index[token])
        
        if canonical_first and len(canonical_first) > 2:
            first_key = f"_FIRST_{canonical_first}"
            if first_key in index:
                candidate_indices.update(index[first_key])
        
        variations = []
        for idx in candidate_indices:
            if idx < len(names):
                name = names[idx]
                score = calculate_similarity(canonical_name, name)
                if score >= threshold:
                    variations.append(name)
        
        all_mappings.append({
            "canonical": canonical_name,
            "variations": variations,
            "count": len(variations)
        })
    
    # Format output
    if output_format == "json":
        return json.dumps(all_mappings, indent=2)
    
    elif output_format == "detailed":
        result = ["## Bulk Canonical Mappings", ""]
        for m in all_mappings:
            result.append(f"### {m['canonical']} ({m['count']} variations)")
            for v in m['variations'][:15]:
                result.append(f"- {v}")
            if m['count'] > 15:
                result.append(f"- ... and {m['count'] - 15} more")
            result.append("")
        return "\n".join(result)
    
    else:  # summary
        result = [
            "## Bulk Mapping Summary",
            "",
            "| Canonical Name | Variations |",
            "|----------------|------------|"
        ]
        total = 0
        for m in all_mappings:
            result.append(f"| {m['canonical']} | {m['count']} |")
            total += m['count']
        result.append("")
        result.append(f"**Total:** {len(all_mappings)} canonical names, {total} variations")
        return "\n".join(result)


tool_registry.register(bulk_create_mappings, "name_matching")


@tool
def get_session_info(session_name: str = "default") -> str:
    """
    Get information about a name matching session.
    
    Args:
        session_name: Name of the session to inspect
    
    Returns:
        Session statistics and sample names
    """
    if session_name not in _name_store:
        available = list(_name_store.keys()) if _name_store else ["none - no sessions created"]
        return f"Session '{session_name}' not found.\n\nAvailable sessions: {', '.join(available)}"
    
    session = _name_store[session_name]
    names = session["names"]
    
    # Calculate statistics
    name_lengths = [len(n) for n in names]
    avg_length = sum(name_lengths) / len(name_lengths) if name_lengths else 0
    
    result = [
        f"## Session: {session_name}",
        "",
        f"**Total Names:** {len(names)}",
        f"**Average Name Length:** {avg_length:.1f} characters",
        f"**Index Tokens:** {len(session['index'])}",
        "",
        "### Sample Names (first 10):",
    ]
    
    for name in names[:10]:
        result.append(f"- {name}")
    
    if len(names) > 10:
        result.append(f"- ... and {len(names) - 10} more")
    
    result.extend([
        "",
        "### Sample Names (random from middle):"
    ])
    
    mid_start = len(names) // 2
    for name in names[mid_start:mid_start+5]:
        result.append(f"- {name}")
    
    return "\n".join(result)


tool_registry.register(get_session_info, "name_matching")


@tool
def clear_session(session_name: str = "default") -> str:
    """
    Clear a name matching session to free memory.
    
    Args:
        session_name: Name of the session to clear, or 'all' to clear all sessions
    
    Returns:
        Confirmation of cleared session(s)
    """
    if session_name == "all":
        count = len(_name_store)
        _name_store.clear()
        return f"✅ Cleared all {count} sessions"
    
    if session_name in _name_store:
        del _name_store[session_name]
        return f"✅ Cleared session '{session_name}'"
    
    return f"Session '{session_name}' not found"


tool_registry.register(clear_session, "name_matching")


@tool
def analyze_name(name: str) -> str:
    """
    Analyze a single name to show how it will be processed for matching.
    Useful for understanding why matches are or aren't found.
    
    Args:
        name: The name to analyze
    
    Returns:
        Analysis showing normalization steps and tokens
    """
    result = [
        f"## Name Analysis: {name}",
        "",
        "### Processing Steps:",
        "",
        f"1. **Original:** {name}",
        f"2. **Normalized:** {normalize_name(name)}",
        f"3. **Without Suffixes:** {remove_suffixes(name)}",
        f"4. **Tokens:** {', '.join(sorted(get_name_tokens(name)))}",
        "",
        "### Abbreviation Expansions Applied:",
    ]
    
    name_upper = name.upper()
    expansions = []
    for abbr, full in COMMON_ABBREVIATIONS.items():
        if abbr in name_upper:
            expansions.append(f"- {abbr} → {full}")
    
    if expansions:
        result.extend(expansions)
    else:
        result.append("- None detected")
    
    result.extend([
        "",
        "### Removed Business Suffixes:",
    ])
    
    removed = []
    for suffix in BUSINESS_SUFFIXES:
        if suffix in name_upper.split():
            removed.append(f"- {suffix}")
    
    if removed:
        result.extend(removed)
    else:
        result.append("- None detected")
    
    return "\n".join(result)


tool_registry.register(analyze_name, "name_matching")


def get_name_matching_tools():
    """Get all name matching tools."""
    return tool_registry.get_tools_by_category("name_matching")
