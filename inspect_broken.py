def main():
    with open('src/pipelines/phasing.rs', 'r') as f:
        content = f.read()

    broken_idx = 89348
    print("--- Around broken part ---")
    print(content[broken_idx-200:broken_idx+500])
    
    start_idx = 102454
    print("\n--- Around start of run_phase_baum_iteration_stage1 ---")
    print(content[start_idx:start_idx+500])

main()
