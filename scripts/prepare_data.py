import os
import sys
import subprocess
import glob

def install_convert_genome():
    """Installs convert_genome using the official install script."""
    print("Installing convert_genome...")
    cmd = "curl -fsSL https://raw.githubusercontent.com/SauersML/convert_genome/main/install.sh | bash"
    subprocess.check_call(cmd, shell=True)
    
    # Add cargo bin to path for this session
    home = os.path.expanduser("~")
    bin_path = os.path.join(home, ".cargo", "bin")
    os.environ["PATH"] += os.pathsep + bin_path
    
    # Verify installation
    try:
        subprocess.check_call(["convert_genome", "--version"])
        print("convert_genome installed successfully.")
    except Exception as e:
        print(f"Error verifying convert_genome: {e}")
        sys.exit(1)

def reconstruct_split_files(file_path):
    """Reconstructs split files (e.g., .part-00, .part-01) if they exist."""
    directory = os.path.dirname(file_path)
    basename = os.path.basename(file_path)
    
    parts = sorted(glob.glob(os.path.join(directory, f"{basename}.part*")))
    if parts:
        print(f"Reconstructing {file_path} from {len(parts)} parts...")
        with open(file_path, 'wb') as outfile:
            for part in parts:
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())
        print("Reconstruction complete.")

def run_conversion(input_path, output_vcf):
    """Runs convert_genome to convert input to VCF."""
    print(f"Converting {input_path} to {output_vcf}...")
    
    # Use remote Chr22 reference
    ref_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
    
    cmd = [
        "convert_genome",
        input_path,
        ref_url,
        output_vcf,
        "--format", "vcf"
    ]
    
    subprocess.check_call(cmd)
    print("Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 prepare_data.py <input_file> <output_vcf>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_vcf = sys.argv[2]
    
    install_convert_genome()
    reconstruct_split_files(input_file)
    run_conversion(input_file, output_vcf)
