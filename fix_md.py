with open("core/1.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
current_line = ""

for line in lines:
    stripped = line.strip()
    if stripped.startswith("|"):
        if current_line:
            new_lines.append(current_line)
        current_line = line.rstrip('\n')
    else:
        # If it doesn't start with |, it's a continuation of the previous line.
        # We append it. If the previous ends with english letter and this starts with one, maybe need space.
        # But just appending stripped with a space is safest.
        if current_line.endswith("-"): 
            current_line += stripped 
        else:
            current_line += " " + stripped

if current_line:
    new_lines.append(current_line)

with open("core/1.md", "w", encoding="utf-8") as f:
    for line in new_lines:
        f.write(line + "\n")
