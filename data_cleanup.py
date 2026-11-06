import re

def load_and_clean_chat(filepath):
    with open(filepath,
            "r", encoding="utf-8") as f:
        raw_text = f.read()

    #clean up
    banned_keywords = [
        "Messages and calls are end-to-end encrypted",
        "AUDIO", 
        "https", 
        ".pdf", 
        ".img", 
        ".png"
    ]

    clean_lines = []

    for line in raw_text.splitlines():
        # 2. Check all banned words in a single, fast line
        if any(keyword in line for keyword in banned_keywords):
            continue

        # 3. Strip the timestamp from the start of the line
        line = re.sub(r"^\[\d{1,2}/\d{1,2}/\d{4},\s.*?\]\s*", "", line)

        line = re.sub(
        r"MAMA.*?:",
        "MAMA:",
        line
        )


        # 4. Save the line if it isn't completely blank
        if line.strip():
            clean_lines.append(line.strip())

    text = "\n".join(clean_lines)
    return text
