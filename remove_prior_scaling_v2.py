import sys
import os

def main():
    filepath = "src/model/impute_hmm.rs"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        sys.exit(1)

    with open(filepath, "r") as f:
        content = f.read()

    # Pattern to find (from my previous restore)
    pattern = "prior_counts[idx] = probs.get(idx).copied().unwrap_or(0.0) * active_states as f32;"

    # Replacement with no scaling
    replacement = "prior_counts[idx] = probs.get(idx).copied().unwrap_or(0.0);"

    if pattern in content:
        print(f"Found {content.count(pattern)} occurrences. Removing '* active_states' scaling...")
        content = content.replace(pattern, replacement)
        with open(filepath, "w") as f:
            f.write(content)
        print("Done.")
    else:
        print("Pattern not found.")

if __name__ == "__main__":
    main()
