import re
import sys

DECISION_KEYWORDS = {
    'if': r'\bif\b',
    'else if': r'\belse\s+if\b',
    'for': r'\bfor\b',
    'while': r'\bwhile\b',
    'case': r'\bcase\b',
    'catch': r'\bcatch\b',
    'ternary (?)': r'\?',
    'and (&&)': r'&&',
    'or (||)': r'\|\|'
}

FUNC_HEADER_PATTERN = re.compile(
    r'^[\w\s\*\&\:\<\>\[\]]+?\b([\w:~]+)\s*\(([^)]*)\)\s*(const)?\s*\{', re.MULTILINE
)

def extract_functions(code):
    """Extracts functions and their bodies using brace tracking."""
    functions = []
    matches = list(FUNC_HEADER_PATTERN.finditer(code))
    
    for i, match in enumerate(matches):
        func_name = match.group(1)
        start_index = match.end() - 1  # Include the opening {
        brace_count = 1
        pos = start_index
        while brace_count > 0 and pos < len(code) - 1:
            pos += 1
            if code[pos] == '{':
                brace_count += 1
            elif code[pos] == '}':
                brace_count -= 1

        func_body = code[start_index:pos+1]
        functions.append((func_name, func_body))

    return functions

def count_decisions(code_block):
    breakdown = {}
    total = 0
    for name, pattern in DECISION_KEYWORDS.items():
        matches = re.findall(pattern, code_block)
        count = len(matches)
        if count > 0:
            breakdown[name] = count
            total += count
    return total, breakdown

def analyze_cuda_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        code = file.read()

    functions = extract_functions(code)
    results = []
    total_complexity = 0

    for func_name, body in functions:
        decision_count, breakdown = count_decisions(body)
        complexity = 1 + decision_count
        total_complexity += complexity

        results.append({
            'name': func_name,
            'complexity': complexity,
            'breakdown': breakdown
        })

    return results, total_complexity

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cyclomatic_cuda.py <file.cu>")
        sys.exit(1)

    file_path = sys.argv[1]
    functions, total = analyze_cuda_file(file_path)

    for func in functions:
        print(f"\n🔍 Function: {func['name']}")
        print(f"   Cyclomatic Complexity: {func['complexity']}")
        print("   Breakdown:")
        for keyword, count in func['breakdown'].items():
            print(f"     {keyword}: {count}")
    
    print("\n🧮 Total Cyclomatic Complexity:", total)
