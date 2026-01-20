import os
import sys
import subprocess
import glob
import shutil

def install_convert_genome():
    """Installs convert_genome using cargo install (avoids GitHub API rate limits)."""
    print("Installing convert_genome...")
    # Only install if not present
    if shutil.which("convert_genome"):
        print("convert_genome already installed.")
        return

    # Use cargo install instead of the bash script to avoid GitHub API rate limits
    try:
        subprocess.check_call(["cargo", "install", "convert_genome"])
    except subprocess.CalledProcessError:
        # Fallback to git-based install if crates.io version is outdated
        subprocess.check_call(["cargo", "install", "--git", "https://github.com/SauersML/convert_genome.git"])
    
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
    """Downloads and prepares HGDP+1kG Chr22 reference (hg38)."""
    if os.path.exists(output_vcf) and os.path.exists(output_vcf + ".csi"):
        print("Reference already exists.")
        return

    print("Downloading HGDP+1kG reference panel (Chr22, hg38)...")
    # This URL is for gnomAD HGDP+1kG v3 (hg38)
    bcf_url = "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf"
    
    # Download BCF
    subprocess.check_call(["wget", "-q", bcf_url, "-O", "ref_raw.bcf"])
    
    # Convert BCF -> VCF.gz and Index
    print("Converting BCF to VCF.gz...")
    subprocess.check_call(f"bcftools view ref_raw.bcf -Oz -o {output_vcf}", shell=True)
    subprocess.check_call(["bcftools", "index", "-f", output_vcf])
    
    # Cleanup
    if os.path.exists("ref_raw.bcf"):
        os.remove("ref_raw.bcf")
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
    needs_conversion = False  # Track if we need to convert from non-VCF format
    raw_file = None  # Path to raw file if conversion needed
    
    # Check for split parts (these are VCF.gz parts)
    parts = sorted(glob.glob(os.path.join(input_dir, "*.vcf.gz.part*")))
    if parts:
        print(f"Found {len(parts)} split VCF parts, combining...")
        # Combine parts then compress properly through bcftools
        combined_raw = "truth_combined_raw.vcf.gz"
        subprocess.check_call(f"cat {os.path.join(input_dir, '*.vcf.gz.part*')} > {combined_raw}", shell=True)
        # Re-compress through bcftools to ensure valid BGZF
        subprocess.check_call(f"bcftools view {combined_raw} -Oz -o {source_vcf}", shell=True)
        os.remove(combined_raw)
    else:
        # Check for zip
        zips = glob.glob(os.path.join(input_dir, "*Christopher*.zip")) + glob.glob(os.path.join(input_dir, "*.zip"))
        if zips:
            print(f"Found zip file: {zips[0]}, extracting...")
            extract_dir = "temp_truth_extract"
            os.makedirs(extract_dir, exist_ok=True)
            subprocess.check_call(["unzip", "-o", zips[0], "-d", extract_dir])
            
            # Find VCF files first
            vcf_candidates = []
            text_candidates = []
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    if f.endswith(".vcf") or f.endswith(".vcf.gz"):
                        vcf_candidates.append(fpath)
                    elif f.endswith(".txt") or f.endswith(".csv") or f.endswith(".tsv"):
                        text_candidates.append(fpath)
            
            if vcf_candidates:
                best = max(vcf_candidates, key=os.path.getsize)
                print(f"Found VCF in zip: {best}")
                if best.endswith(".vcf.gz"):
                    # Re-compress through bcftools to ensure valid BGZF
                    subprocess.check_call(f"bcftools view {best} -Oz -o {source_vcf}", shell=True)
                else:
                    # Plain VCF - compress with bcftools
                    subprocess.check_call(f"bcftools view {best} -Oz -o {source_vcf}", shell=True)
            elif text_candidates:
                # No VCF found - this is a genotyping array text file, needs conversion
                best = max(text_candidates, key=os.path.getsize)
                print(f"No VCF found, found text file that needs conversion: {best}")
                needs_conversion = True
                raw_file = best
            else:
                # Last resort - largest file
                all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(extract_dir) for f in filenames]
                if all_files:
                    best = max(all_files, key=os.path.getsize)
                    print(f"No VCF or text found, largest file needs conversion: {best}")
                    needs_conversion = True
                    raw_file = best
                else:
                    shutil.rmtree(extract_dir)
                    raise FileNotFoundError("Zip file was empty")
            
            if not needs_conversion:
                shutil.rmtree(extract_dir)
        else:
            # Fallback - look for existing VCF
            vcfs = glob.glob(os.path.join(input_dir, "*.vcf.gz"))
            if vcfs:
                # Re-compress through bcftools to ensure valid BGZF
                subprocess.check_call(f"bcftools view {vcfs[0]} -Oz -o {source_vcf}", shell=True)
            else:
                raise FileNotFoundError("Could not find Truth VCF source files")
    
    # If we need to convert from a text format (23andMe, AncestryDNA, etc.)
    if needs_conversion:
        print(f"Converting {raw_file} to VCF format using convert_genome...")
        install_convert_genome()
        
        # Use hg19/GRCh37 reference to match 1000G Phase 3 reference panel
        ref_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr22.fa.gz"
        temp_vcf = "temp_truth_conv.vcf"
        
        cmd = ["convert_genome", raw_file, ref_url, temp_vcf, "--format", "vcf"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        if not os.path.exists(temp_vcf):
            raise RuntimeError("convert_genome failed to produce output VCF")
        
        # Compress with bcftools for proper BGZF
        subprocess.check_call(f"bcftools view {temp_vcf} -Oz -o {source_vcf}", shell=True)
        os.remove(temp_vcf)
        
        # Cleanup extract dir if it exists
        if os.path.exists("temp_truth_extract"):
            shutil.rmtree("temp_truth_extract")

    # Index before filtering (required for filtering regions efficiently)
    print("Indexing Truth VCF...")
    subprocess.check_call(["bcftools", "index", "-t", source_vcf])

    # Rename chroms (22 -> chr22) to match reference panel which uses chr22 notation
    with open("chr_map.txt", "w") as f:
        f.write("22\tchr22\n")

    print("Filtering Truth to Chr22...")
    # Filter FIRST using index (regions 22 or chr22), then rename to chr22
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
    """Runs convert_genome to convert input to VCF (hg19) then lifts to hg38."""
    
    # Pre-process input (handle zip/split)
    raw_file = prepare_input_file(input_path)
    
    print(f"Converting {raw_file} to GRCh38 VCF...")

    ref_hg38_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
    temp_hg38_vcf = "temp_hg38.vcf"

    cmd = [
        "convert_genome",
        raw_file,
        ref_hg38_url,
        temp_hg38_vcf,
        "--assembly", "GRCh38",
        "--format", "vcf",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    if not os.path.exists(temp_hg38_vcf):
        raise RuntimeError("convert_genome failed to produce temp_hg38.vcf")

    print("Finalizing GRCh38 output...")
    print("Filtering invalid records (missing ALT but non-ref GT)...")
    filter_cmd = f"bcftools view {temp_hg38_vcf} -e 'ALT=\".\" && GT[*]=\"alt\"' -Oz -o {output_vcf}"
    subprocess.check_call(filter_cmd, shell=True)
    subprocess.check_call(["bcftools", "index", "-f", output_vcf])
    print("Conversion complete.")

    # Cleanup
    if os.path.exists("temp_hg38.vcf"):
        os.remove("temp_hg38.vcf")
        
    # Cleanup extracted if it was temp
    if "extracted" in raw_file:
        shutil.rmtree(os.path.dirname(raw_file))
        
    print("Conversion preparation complete.")

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
