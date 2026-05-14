import re
import glob

pattern = "activated_*.log"

overall_activated = 0.0
overall_tokens = 0

file_means = []

for log_file in sorted(glob.glob(pattern)):
    file_activated = 0.0
    file_tokens = 0

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(
                r"average activated:\s*([0-9.]+),\s*num_tokens:\s*([0-9]+)",
                line
            )
            if match:
                act = float(match.group(1))
                tokens = int(match.group(2))

                file_activated += act * tokens
                file_tokens += tokens

    if file_tokens > 0:
        file_mean = file_activated / file_tokens
        file_means.append(file_mean)

        print(f"{log_file}: mean={file_mean:.4f}, tokens={file_tokens}")

        overall_activated += file_activated
        overall_tokens += file_tokens
    else:
        print(f"{log_file}: no valid data")


if file_means:
    cross_file_mean = sum(file_means) / len(file_means)
    print("\nCross-file mean:")
    print(f"{cross_file_mean:.4f}")