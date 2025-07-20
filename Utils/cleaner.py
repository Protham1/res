import json
import re

def clean_raw_text(text: str) -> str:
    """Clean raw resume text for token efficiency and GPT processing"""
    
    # Normalize line endings and remove extra spaces
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse multiple newlines
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Remove bullet points or special Unicode characters
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\*•→·\-]+', '-', text)

    # Remove LaTeX-style junk if any
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)

    # Fix common formatting
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)  # Remove lone newlines between words

    # Ensure section headers are spaced
    headers = ['education', 'skills', 'experience', 'projects', 'contact', 'summary', 'certifications']
    for header in headers:
        pattern = rf'(?i)\b{header}\b'
        text = re.sub(pattern, lambda m: '\n\n' + m.group(0).title(), text, flags=re.IGNORECASE)

    # Strip extra whitespace
    return text.strip()


def load_and_clean_json(path: str, save_cleaned: bool = False):
    """Load JSON resume, clean raw_text, optionally save"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_text = data.get("raw_text", "")
    if not raw_text:
        raise ValueError("No 'raw_text' field found in JSON")

    cleaned = clean_raw_text(raw_text)
    print("\n🔹 Cleaned Resume Text:\n")
    print(cleaned[:1500], "..." if len(cleaned) > 1500 else "")

    if save_cleaned:
        data['cleaned_text'] = cleaned
        with open("cleaned_resume.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("\n✅ Cleaned text saved to 'cleaned_resume.json'")

    return cleaned

# Example usage
if __name__ == "__main__":
    file_path = "resume_raw_text.json"  # <-- your JSON path
    load_and_clean_json(file_path, save_cleaned=True)
