import os
import sys
import subprocess
import glob
import shutil

def install_convert_genome():
    """Installs convert_genome using the official install script."""
    print("Installing convert_genome...")
    # Only install if not present
    if shutil.which("convert_genome"):
        print("convert_genome already installed.")
        return

    cmd = "curl -fsSL https://raw.githubusercontent.com/SauersML/convert_genome/main/install.sh | bash"
    subprocess.check_call(cmd, shell=True)
    
    # Add cargo bin to path for this session
    home = os.path.expanduser("~")
    bin_path = os.path.join(home, ".cargo", "bin")
    os.environ["PATH"] += os.pathsep + bin_path

def prepare_input_file(input_path):
    """
    Prepares the input file for conversion.
    - If split parts exist, combines them.
    - If zip file, extracts it.
    - Returns the path to the actual raw data file.
    """
    directory = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    
    # 1. Handle Split Files
    parts = sorted(glob.glob(os.path.join(directory, f"{basename}.part*")))
    if parts:
        combined_path = input_path.replace(".part-00", "") # Heuristic
        if not combined_path.endswith(".txt") and not combined_path.endswith(".csv"):
             combined_path = os.path.join(directory, "combined_input.txt")
             
        print(f"Detected split files. Combining {len(parts)} parts to {combined_path}...")
        # Fast shell concat
        subprocess.check_call(f"cat '{input_path}'.part* > '{combined_path}'", shell=True)
        return combined_path

    # 2. Handle Zip Files
    if input_path.endswith(".zip"):
        print(f"Detected zip file: {input_path}")
        extract_dir = os.path.join(directory, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        print("Unzipping...")
        subprocess.check_call(["unzip", "-o", input_path, "-d", extract_dir])
        
        # Find the data file inside
        # Look for txt or csv or large files
        candidates = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.endswith(".txt") or f.endswith(".csv") or f.endswith(".tsv"):
                    candidates.append(os.path.join(root, f))
        
        if not candidates:
            # Maybe it's a file without extension? Take the largest one.
            all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(extract_dir) for f in filenames]
            if all_files:
                largest = max(all_files, key=os.path.getsize)
                print(f"No obvious text file found, using largest file: {largest}")
                return largest
            else:
                raise ValueError("Zip file appeared empty.")
        
        # Heuristic: pick the largest text file
        target = max(candidates, key=os.path.getsize)
        print(f"Using extracted file: {target}")
        return target

    return input_path

def download_reference(output_vcf):
    """Downloads and prepares 1000G Chr22 reference."""
    if os.path.exists(output_vcf) and os.path.exists(output_vcf + ".tbi"):
        print("Reference already exists.")
        return

    print("Downloading reference panel (Chr22)...")
    url = "https://s3.amazonaws.com/1000genomes/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
    
    # Download
    subprocess.check_call(["wget", "-q", url, "-O", "ref_raw.vcf.gz"])
    
    # Filter/Normalize
    print("Filtering reference...")
    # It is already Chr22, just normalize variants
    # No regions filter needed if file is just Chr22 (avoids index issue on stream if using pipe, but here we use file)
    # Using -m 2 -M 2 -v snps to keep only biallelic SNPs
    subprocess.check_call("bcftools view ref_raw.vcf.gz -m 2 -M 2 -v snps -Oz -o " + output_vcf, shell=True)
    subprocess.check_call(["tabix", "-f", output_vcf])
    
    # Cleanup
    if os.path.exists("ref_raw.vcf.gz"):
        os.remove("ref_raw.vcf.gz")
    print(f"Reference prepared: {output_vcf}")

def prepare_truth(source, output_vcf):
    """Reconstructs and prepares Truth VCF (Chr22). Source can be dir or person name."""
    if source.lower() == "kat":
        input_dir = "data/kat_suricata"
    elif source.lower() == "christopher":
        input_dir = "data/christopher_smith"
    else:
        input_dir = source

    print(f"Preparing Truth VCF from {input_dir}...")
    
    source_vcf = "truth_full.vcf.gz"
    
    # Check for split parts
    parts = sorted(glob.glob(os.path.join(input_dir, "*.part*")))
    if parts:
        print("Found split parts, combining...")
        subprocess.check_call(f"cat {os.path.join(input_dir, '*.part*')} > {source_vcf}", shell=True)
    else:
        # Check for zip
        zips = glob.glob(os.path.join(input_dir, "*Christopher*.zip"))
        if zips:
            print(f"Found zip file: {zips[0]}, extracting...")
            # Extract to a temp dir to find the VCF safely
            extract_dir = "temp_truth_extract"
            os.makedirs(extract_dir, exist_ok=True)
            subprocess.check_call(["unzip", "-o", zips[0], "-d", extract_dir])
            
            # Find largest vcf or vcf.gz
            candidates = []
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    if f.endswith(".vcf") or f.endswith(".vcf.gz"):
                        candidates.append(os.path.join(root, f))
            
            if candidates:
                best = max(candidates, key=os.path.getsize)
                print(f"Found VCF in zip: {best}")
                os.rename(best, source_vcf)
            else:
                # If no VCF extension, try largest file
                all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(extract_dir) for f in filenames]
                if all_files:
                    best = max(all_files, key=os.path.getsize)
                    print(f"No .vcf found, using largest file as VCF: {best}")
                    os.rename(best, source_vcf)
                else:
                    raise FileNotFoundError("Zip file was empty")
            
            shutil.rmtree(extract_dir)
        else:
            # Fallback
            vcfs = glob.glob(os.path.join(input_dir, "*.vcf.gz"))
            if vcfs:
                subprocess.check_call(["cp", vcfs[0], source_vcf])
            else:
                raise FileNotFoundError("Could not find Truth VCF source files")

    # Index before filtering (required for filtering regions efficiently)
    print("Indexing Truth VCF...")
    subprocess.check_call(["bcftools", "index", "-t", source_vcf])

    # Rename chroms (chr22 -> 22) and filter
    with open("chr_map.txt", "w") as f:
        f.write("chr22\t22\n")

    print("Filtering Truth to Chr22...")
    # Filter FIRST using index (regions 22 or chr22), then rename
    cmd = (
        f"bcftools view {source_vcf} --regions 22,chr22 -Ou | "
        f"bcftools annotate --rename-chrs chr_map.txt -Oz -o {output_vcf}"
    )
    subprocess.check_call(cmd, shell=True)
    subprocess.check_call(["tabix", "-p", "vcf", output_vcf])
    
    if os.path.exists(source_vcf):
        os.remove(source_vcf)
    if os.path.exists("chr_map.txt"):
        os.remove("chr_map.txt")
    print(f"Truth prepared: {output_vcf}")

def run_conversion(input_path, output_vcf):
    """Runs convert_genome to convert input to VCF."""
    
    # Pre-process input (handle zip/split)
    raw_file = prepare_input_file(input_path)
    
    # Rename chromosomes from chr22 -> 22 to match 1000G reference
    # Create a mapping file
    with open("chr_map.txt", "w") as f:
        f.write("chr22\t22\n")

    print(f"Converting {raw_file} to {output_vcf}...")
    
    # Use remote Chr22 reference for standardization
    ref_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
    
    # convert_genome uses positional arguments: <INPUT> <REFERENCE> [OUTPUT]
    cmd = [
        "convert_genome",
        raw_file,
        ref_url,
        "temp_conv.vcf",
        "--format", "vcf"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    if not os.path.exists("temp_conv.vcf"):
        raise RuntimeError("convert_genome failed to produce temp_conv.vcf")
        
    print(f"temp_conv.vcf size: {os.path.getsize('temp_conv.vcf')}")
    
    # Compress, normalize chroms, filter to Chr22
    # convert_genome output is already filtered to Chr22 by reference
    # We pipe: view -> annotate (rename) -> output
    # Removed --regions 22 to avoid index requirement on stream
    
    cmd_process = (
        f"bcftools view temp_conv.vcf -Ou | "
        f"bcftools annotate --rename-chrs chr_map.txt -Oz -o {output_vcf}"
    )
    subprocess.check_call(cmd_process, shell=True)
    subprocess.check_call(["tabix", "-p", "vcf", output_vcf])
    
    if os.path.exists("temp_conv.vcf"):
        os.remove("temp_conv.vcf")
    if os.path.exists("chr_map.txt"):
        os.remove("chr_map.txt")
        
    # Cleanup extracted if it was temp
    if "extracted" in raw_file:
        shutil.rmtree(os.path.dirname(raw_file))
        
    print("Conversion complete.")

if __name__ == "__main__":
    print("Prepare Data Script v1.2 (Clean Overwrite)")
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 prepare_data.py array <input_file> <output_vcf>")
        print("  python3 prepare_data.py reference <output_vcf>")
        print("  python3 prepare_data.py truth <person_or_dir> <output_vcf>")
        sys.exit(1)
        
    mode = sys.argv[1]
    
    if mode == "array":
        install_convert_genome()
        run_conversion(sys.argv[2], sys.argv[3])
        # Post-process array to Chr22 (redundant if run_conversion does it, but kept for safety if run_conversion changed)
        # Actually run_conversion now does everything including filtering to 22.
        # So we just verify.
        if not os.path.exists(sys.argv[3]):
             print(f"Error: {sys.argv[3]} was not created.")
             sys.exit(1)
        
    elif mode == "reference":
        download_reference(sys.argv[2])
        
    elif mode == "truth":
        prepare_truth(sys.argv[2], sys.argv[3])
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
