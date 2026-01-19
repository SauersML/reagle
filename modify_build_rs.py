import sys
import re
import os

def main():
    filepath = "build.rs"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    with open(filepath, "r") as f:
        content = f.read()

    # 1. Remove call in main
    # We construct a regex that matches the specific call block including the comment and update_stage call.
    # We use re.DOTALL to let . match newlines if needed, but the structure suggests mostly single lines.
    # However, to be safe with indentation:
    pattern = r'(\s*)// Manually check for unused variables in the build script\s+update_stage\("manual lint self-check"\);\s+manually_check_for_unused_variables\(\);'
    
    if re.search(pattern, content):
        content = re.sub(pattern, '', content)
        print("Removed manually_check_for_unused_variables call in main.")
    else:
        print("Warning: Could not find manually_check_for_unused_variables call in main.")

    # 2. Remove function definitions.
    lines = content.splitlines()
    new_lines = []
    skip = False
    brace_count = 0
    
    # Signatures to look for (trimmed)
    funcs_to_remove = [
        "fn manually_check_for_unused_variables() {",
        "fn manual_lint_arguments(build_path: &Path) -> Vec<OsString> {",
        "fn build_dependencies_directory() -> Option<PathBuf> {",
        "fn locate_build_dependency(deps_dir: &Path, crate_name: &str) -> Option<PathBuf> {",
        "fn command_preview(program: &OsStr, args: &[OsString]) -> String {",
    ]
    
    removed_funcs = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for function start
        found_start = False
        for func_start in funcs_to_remove:
            if stripped == func_start or (stripped.startswith("fn ") and func_start in stripped):
                 # Double check exact signature match or close enough
                 if func_start in line:
                    skip = True
                    # Initialize brace count based on this line
                    brace_count = line.count('{') - line.count('}')
                    found_start = True
                    removed_funcs += 1
                    break
        
        if found_start:
            # If there was a specific comment for the function right before, strictly we leave it or remove it.
            # But the logic here only removes the function body.
            i += 1
            continue
            
        if skip:
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count == 0:
                skip = False
            i += 1
            continue
            
        # Update import
        if "use std::ffi::{OsStr, OsString};" in line:
            new_lines.append(line.replace("use std::ffi::{OsStr, OsString};", "use std::ffi::OsStr;"))
        else:
            new_lines.append(line)
        i += 1
        
    print(f"Removed {removed_funcs} functions.")

    with open(filepath, "w") as f:
        f.write("\n".join(new_lines) + "\n")

if __name__ == "__main__":
    main()
