import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from fpdf import FPDF

# === CONFIGURATION ===
LOG_DIR = "logs"  # your TensorBoard logs directory
OUTPUT_DIR = "analysis_outputs"
TAGS = [
    "train/loss",
    "train/learning_rate",
    "eval/loss",
    "eval/accuracy",
    "eval/f1",
    "eval/precision",
    "eval/recall",
]


# === FUNCTION TO EXTRACT SCALARS FROM ONE EVENT FILE ===
def extract_scalars_from_eventfile(event_file, tags):
    ea = EventAccumulator(
        event_file,
        size_guidance={  # limits number of items per tag loaded
            "scalars": 0,  # 0 means load all
        },
    )
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    scalars = {}
    for tag in tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            if events:
                scalars[tag] = pd.DataFrame(
                    {
                        "step": [e.step for e in events],
                        "value": [e.value for e in events],
                    }
                )
    return scalars


# === FIND ALL EVENT FILES ===
def find_event_files(log_dir):
    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, f))
    return event_files


# === PLOT AND SAVE GRAPHS ===
def plot_scalars(all_scalars, output_dir, tags):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    for tag in tags:
        plt.figure(figsize=(10, 5))
        plotted = False
        for event_file, scalars in all_scalars.items():
            if tag in scalars:
                df = scalars[tag]
                if not df.empty:
                    sns.lineplot(
                        x="step", y="value", data=df, label=os.path.basename(event_file)
                    )
                    plotted = True
        if plotted:
            plt.title(f"{tag} over Training Steps")
            plt.xlabel("Step")
            plt.ylabel(tag)
            plt.legend()
            plt.grid(True)

            # Create directory for this tag's plot if nested
            # get the directory path for this tag's image
            dir_for_tag = os.path.dirname(os.path.join(output_dir, f"{tag}.png"))
            os.makedirs(dir_for_tag, exist_ok=True)

            plot_path = os.path.join(output_dir, f"{tag}.png")
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
        plt.close()
    return plot_paths


# === PDF REPORT CLASS ===
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Training Metrics Report", ln=True, align="C")
        self.ln(10)

    def add_plot(self, image_path, title):
        self.add_page()
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True, align="C")
        self.ln(10)
        self.image(image_path, x=15, y=30, w=180)


# === MAIN EXECUTION ===
def main():
    print(f"Looking for TensorBoard event files in '{LOG_DIR}'...")
    event_files = find_event_files(LOG_DIR)
    if not event_files:
        print("❌ No TensorBoard event files found.")
        return

    print(f"Found {len(event_files)} event files. Extracting scalars...")
    all_scalars = {}
    for ef in event_files:
        scalars = extract_scalars_from_eventfile(ef, TAGS)
        print(f"Extracted tags from {os.path.basename(ef)}: {list(scalars.keys())}")
        if scalars:
            all_scalars[ef] = scalars

    if not all_scalars:
        print("❌ No scalar data extracted. Check your log files and TAGS.")
        return

    print("Plotting scalars...")
    plot_paths = plot_scalars(all_scalars, OUTPUT_DIR, TAGS)

    if not plot_paths:
        print("❌ No plots generated.")
        return

    print(f"Generating PDF report with {len(plot_paths)} plots...")
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)

    for path in plot_paths:
        tag_name = os.path.basename(path).replace(".png", "").replace("_", " ").title()
        pdf.add_plot(path, tag_name)

    report_path = os.path.join(OUTPUT_DIR, "training_metrics_report.pdf")
    pdf.output(report_path)
    print(f"✅ PDF report generated at: {report_path}")


if __name__ == "__main__":
    main()
