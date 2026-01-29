import re

def main():
    with open('src/pipelines/phasing.rs', 'r') as f:
        content = f.read()

    # Find the start of run_phase_baum_iteration_stage1
    start_match = re.search(r'fn run_phase_baum_iteration_stage1\s*\(', content)
    if not start_match:
        print("Could not find start of run_phase_baum_iteration_stage1")
        return

    start_idx = start_match.start()
    
    # Find the matching closing brace for the function
    open_braces = 0
    end_idx = -1
    for i in range(start_idx, len(content)):
        if content[i] == '{':
            open_braces += 1
        elif content[i] == '}':
            open_braces -= 1
            if open_braces == 0:
                end_idx = i + 1
                break
    
    if end_idx == -1:
        print("Could not find end of run_phase_baum_iteration_stage1 - likely syntax error from previous edit")
        # Fallback: scan for the next function or implementation block?
        # Since we know the file has syntax errors, the brace counting might fail if the error is inside.
        # But the error reported earlier was 'mismatched closing delimiter', so likely we have too many or too few braces.
        
        # Let's verify the content around the reported error.
        # error: mismatched closing delimiter: `}`
        # --> src/pipelines/phasing.rs:2746:60
        
        # The line numbers from the error message might be useful.
        pass

    print(f"Function starts at index {start_idx}")
    if end_idx != -1:
        print(f"Function ends at index {end_idx}")
        # print snippet
        print(content[start_idx:start_idx+200])
        print("...")
        print(content[end_idx-50:end_idx])
    
    # Check for the broken part
    broken_part_idx = content.find('tracing::info_span!("hmm_samples").in_scope(|| {')
    if broken_part_idx != -1:
        print(f"Found broken part at {broken_part_idx}")

main()
