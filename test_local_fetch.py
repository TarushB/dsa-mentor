import re
def get_local_description(title):
    try:
        with open('data/output_leetcode_questions.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # The title might have trailing spaces in the file
        search_title = re.escape(title.strip().lower())
        # Look for:
        # ------------------------------
        # {Title} (possibly with trailing spaces)
        # ------------------------------
        # (Description until the next --- or EOF)
        
        pattern = re.compile(
            r'-{20,}\n\s*(' + search_title + r')\s*\n-{20,}\n(.*?)(?=\n-{20,}|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        
        match = pattern.search(content)
        if match:
            return match.group(2).strip()
    except Exception as e:
        print("Error:", e)
    return None

print(repr(get_local_description('Two Sum')[:50]))
