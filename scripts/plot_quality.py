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

def plot_calibration(metrics, title):
    bins = metrics['calibration']['bins']
    observed = metrics['calibration']['observed_frequencies']
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.plot(bins, observed, 'o-', linewidth=2, label='Model')
    ax.set_title(f'Calibration: {title}')
    ax.grid(True, alpha=0.3)
    return create_plot_image(fig)

def plot_sen_distribution(metrics, title):
    sen_scores = metrics['sen_distribution']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sen_scores, bins=30, color='skyblue', alpha=0.7)
    mean_sen = sum(sen_scores) / len(sen_scores) if sen_scores else 0
    ax.axvline(mean_sen, color='red', linestyle='--', label=f'Mean: {mean_sen:.4f}')
    ax.set_title(f'SEN Dist: {title}')
    ax.legend()
    return create_plot_image(fig)

def generate_reports(data_dir):
    r_data = load_metrics(os.path.join(data_dir, "reagle_metrics.json"))
    b_data = load_metrics(os.path.join(data_dir, "beagle_metrics.json"))
    
    # Generate Plot Data
    r_cal = plot_calibration(r_data, "Reagle")
    b_cal = plot_calibration(b_data, "Beagle")
    r_sen = plot_sen_distribution(r_data, "Reagle")
    b_sen = plot_sen_distribution(b_data, "Beagle")
    
    rows = []
    for tool, m in [("Reagle", r_data), ("Beagle", b_data)]:
        rows.append(f"| {tool} | {m['overall_sen']:.5f} | {m['overall_concordance']:.5f} | {m['n_sites']} |")

    # Combined Markdown Summary (for GHA UI)
    md = f"""
### Imputation Quality: {os.path.basename(data_dir)}

| Tool | Mean SEN | Concordance | Sites |
|---|---|---|---|
{chr(10).join(rows)}

#### Calibration & SEN Distribution
*Note: Plots may not render in Job Summary. Check HTML artifacts.*
"""
    with open(os.path.join(data_dir, "summary.md"), 'w') as f:
        f.write(md)
    
    # Full HTML Report
    html_rows = []
    for tool, m in [("Reagle", r_data), ("Beagle", b_data)]:
        html_rows.append(f"<tr><td>{tool}</td><td>{m['overall_sen']:.5f}</td><td>{m['overall_concordance']:.5f}</td><td>{m['n_sites']}</td></tr>")

    html = f"""
    <html>
    <head><title>Benchmark Report</title><style>table {{ border-collapse: collapse; }} td, th {{ border: 1px solid black; padding: 8px; }}</style></head>
    <body>
    <h1>Imputation Quality: {os.path.basename(data_dir)}</h1>
    <table><tr><th>Tool</th><th>Mean SEN</th><th>Concordance</th><th>Sites</th></tr>
    {''.join(html_rows)}
    </table>
    <h2>Calibration</h2>
    <img src="{r_cal}"> <img src="{b_cal}">
    <h2>SEN Distribution</h2>
    <img src="{r_sen}"> <img src="{b_sen}">
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
