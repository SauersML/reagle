import sys
import re

with open("build.rs", "r") as f:
    content = f.read()

# Define patterns to remove. We'll find the start of manually_check_for_unused_variables
# and remove everything until scan_for_underscore_prefixes starts.

start_marker = "// This function manually checks for unused variables in the current file"
end_marker = "fn scan_for_underscore_prefixes() -> Vec<String> {"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1:
    print("Start marker not found")
    sys.exit(1)

if end_idx == -1:
    print("End marker not found")
    sys.exit(1)

new_content = content[:start_idx] + content[end_idx:]

with open("build.rs", "w") as f:
    f.write(new_content)

print("Successfully removed unused functions.")
