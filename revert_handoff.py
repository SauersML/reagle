import sys
import os

def main():
    filepath = "src/pipelines/imputation_streaming.rs"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        sys.exit(1)

    with open(filepath, "r") as f:
        content = f.read()

    # The block to remove (two-pass loop)
    # I'll search for the comments I added or structure
    start_marker = "// 1. Determine handoff marker index (max of all samples)"
    # End of the block
    # It ends after the second loop.
    # The second loop ends with "}" and then "all_results.sort_by_key..."?
    # No, check `sed` output from earlier or context.

    # I'll replace the entire section with the original loop logic.
    # The original loop logic (reconstructed from earlier grep/sed):
    original_logic = """        for mut item in sample_results {
            if let Some((p1, p2)) = item.priors.take() {
                let base = item.result.sample_idx * 2;
                if base + 1 < next_priors_vec.len() {
                    next_priors_vec[base] = p1;
                    next_priors_vec[base + 1] = p2;
                }
            }
            if let Some(idx) = item.last_info_idx {
                handoff_marker_idx = Some(match handoff_marker_idx {
                    Some(prev) => prev.max(idx),
                    None => idx,
                });
            }

            // Also need to push to all_results (since sample_results is consumed)
            // Wait, original code pushed to all_results?
            // "all_results.push(item.result);"
            // And handled sm_alt_probs?
            // "if (need1 || need2) ..."

            // I need to be careful to restore the FULL original loop including sm_alt_probs.
        }"""

    # Actually, it's safer to read the current file and target the specific 2-pass block I added.

    # Locate start of 2-pass block
    idx_start = content.find(start_marker)
    if idx_start == -1:
        print("Could not find 2-pass block start.")
        return

    # Find the end of the loop.
    # The block ends before "all_results.sort_by_key".
    idx_end = content.find("all_results.sort_by_key")
    if idx_end == -1:
        print("Could not find end of block.")
        return

    # Extract the block to verify
    block_to_replace = content[idx_start:idx_end]
    # print("Replacing block:\n", block_to_replace[:200])

    # Construct the original single loop.
    # I need to make sure I include the sm_alt_probs logic which was inside the loop.
    # In my 2-pass version, I had "populate and decay priors" loop which included sm_alt_probs.
    # So I can just take the body of that loop, remove the decay logic, and put back the simple prior assignment.

    # The structure of the new loop:
    new_loop = """        for mut item in sample_results {
            let sample_idx = item.result.sample_idx;
            let h1 = sample_idx * 2;
            let h2 = h1 + 1;
            let need1 = sm_alt_probs_by_hap
                .get(h1)
                .and_then(|v| v.as_ref())
                .is_some();
            let need2 = sm_alt_probs_by_hap
                .get(h2)
                .and_then(|v| v.as_ref())
                .is_some();
            if (need1 || need2) && output_markers > 0 {
                let p1 = sm_alt_probs_by_hap
                    .get_mut(h1)
                    .and_then(|v| v.take())
                    .unwrap_or_default();
                let p2 = sm_alt_probs_by_hap
                    .get_mut(h2)
                    .and_then(|v| v.take())
                    .unwrap_or_default();
                if need1 {
                    item.result.hap_alt_probs.0 = Some(p1);
                }
                if need2 {
                    item.result.hap_alt_probs.1 = Some(p2);
                }
            }
            all_results.push(item.result);
            if let Some((p1, p2)) = item.priors.take() {
                let base = sample_idx * 2;
                if base + 1 < next_priors_vec.len() {
                    next_priors_vec[base] = p1;
                    next_priors_vec[base + 1] = p2;
                }
            }
            if let Some(idx) = item.last_info_idx {
                handoff_marker_idx = Some(match handoff_marker_idx {
                    Some(prev) => prev.max(idx),
                    None => idx,
                });
            }
        }

        """

    # Note: `target_handoff_idx` variable calculation should be removed too.

    new_content = content[:idx_start] + new_loop + content[idx_end:]

    # Clean up any trailing "let target_handoff_idx..." if it was outside the block?
    # In my 2-pass implementation:
    # // 1. Determine...
    # for ...
    # let target_handoff_idx = ...
    # // 2. Populate...
    # for ...

    # Since I replace from Start Marker to End, and Start Marker was before the first loop,
    # it should cover everything.

    with open(filepath, "w") as f:
        f.write(new_content)
    print("Reverted handoff loop logic.")

if __name__ == "__main__":
    main()
