import sys
import json
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

def load_metrics(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_plot_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def plot_r2_by_maf(metrics, title):
    """Plot R² by MAF bin if available."""
    by_maf = metrics.get('by_maf', {})
    if not by_maf:
        return None
    
    bins = []
    r2_values = []
    for maf_bin, data in sorted(by_maf.items()):
        if data.get('r_squared') is not None:
            bins.append(maf_bin)
            r2_values.append(data['r_squared'])
    
    if not bins:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(bins)), r2_values, color='skyblue', alpha=0.7)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=45, ha='right')
    ax.set_ylabel('R²')
    ax.set_title(f'R² by MAF: {title}')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return create_plot_image(fig)

def plot_concordance_by_maf(metrics, title):
    """Plot concordance by MAF bin if available."""
    by_maf = metrics.get('by_maf', {})
    if not by_maf:
        return None
    
    bins = []
    conc_values = []
    for maf_bin, data in sorted(by_maf.items()):
        if data.get('concordance') is not None:
            bins.append(maf_bin)
            conc_values.append(data['concordance'])
    
    if not bins:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(bins)), conc_values, color='lightcoral', alpha=0.7)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=45, ha='right')
    ax.set_ylabel('Concordance')
    ax.set_ylim([0, 1])
    ax.set_title(f'Concordance by MAF: {title}')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return create_plot_image(fig)

def generate_reports(data_dir):
    r_data = load_metrics(os.path.join(data_dir, "reagle_metrics.json"))
    b_data = load_metrics(os.path.join(data_dir, "beagle_metrics.json"))
    
    # Generate Plot Data (optional - gracefully handle missing)
    r_r2_plot = plot_r2_by_maf(r_data, "Reagle")
    b_r2_plot = plot_r2_by_maf(b_data, "Beagle")
    r_conc_plot = plot_concordance_by_maf(r_data, "Reagle")
    b_conc_plot = plot_concordance_by_maf(b_data, "Beagle")
    
    # Extract metrics with proper field names
    rows = []
    for tool, m in [("Reagle", r_data), ("Beagle", b_data)]:
        sen = m.get('sen_mean')
        sen_str = f"{sen:.5f}" if sen is not None else "N/A"
        
        concordance = m.get('unphased_concordance')
        conc_str = f"{concordance:.5f}" if concordance is not None else "N/A"
        
        sites = m.get('sites_compared', 0)
        
        rows.append(f"| {tool} | {sen_str} | {conc_str} | {sites} |")

    # Combined Markdown Summary (for GHA UI)
    md = f"""
### Imputation Quality: {os.path.basename(data_dir)}

| Tool | Mean SEN | Concordance | Sites |
|---|---|---|---|
{chr(10).join(rows)}

#### Metrics by MAF Bin
*Note: Plots may not render in Job Summary. Check HTML artifacts.*
"""
    with open(os.path.join(data_dir, "summary.md"), 'w') as f:
        f.write(md)
    
    # Full HTML Report
    html_rows = []
    for tool, m in [("Reagle", r_data), ("Beagle", b_data)]:
        sen = m.get('sen_mean')
        sen_str = f"{sen:.5f}" if sen is not None else "N/A"
        
        concordance = m.get('unphased_concordance')
        conc_str = f"{concordance:.5f}" if concordance is not None else "N/A"
        
        sites = m.get('sites_compared', 0)
        
        html_rows.append(f"<tr><td>{tool}</td><td>{sen_str}</td><td>{conc_str}</td><td>{sites}</td></tr>")

    # Build plots section
    plots_html = ""
    if r_r2_plot and b_r2_plot:
        plots_html += f"""
    <h2>R² by MAF Bin</h2>
    <img src="{r_r2_plot}"> <img src="{b_r2_plot}">
    """
    if r_conc_plot and b_conc_plot:
        plots_html += f"""
    <h2>Concordance by MAF Bin</h2>
    <img src="{r_conc_plot}"> <img src="{b_conc_plot}">
    """

    html = f"""
    <html>
    <head><title>Benchmark Report</title><style>table {{ border-collapse: collapse; }} td, th {{ border: 1px solid black; padding: 8px; }}</style></head>
    <body>
    <h1>Imputation Quality: {os.path.basename(data_dir)}</h1>
    <table><tr><th>Tool</th><th>Mean SEN</th><th>Concordance</th><th>Sites</th></tr>
    {''.join(html_rows)}
    </table>
    {plots_html}
    </body></html>
    """
    with open(os.path.join(data_dir, "report.html"), 'w') as f:
        f.write(html)
    
    print(f"Reports generated in {data_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/plot_quality.py <data_dir>")
        sys.exit(1)
    generate_reports(sys.argv[1])
