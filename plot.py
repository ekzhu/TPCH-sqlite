import json
import numpy as np
import matplotlib.pyplot as plt

estimate_files = [
    "estimates-gpt-4-2.json",
    "estimates-gpt-4-3.json",
    "estimates-gpt-4-4.json",
    "estimates-gpt-4-5.json",
    "estimates-gpt-4-6.json",
    "estimates-gpt-4-7.json",
    "estimates-gpt-4-8.json",
    "estimates-gpt-4-9.json",
    "estimates-gpt-4-10.json",
    "estimates-gpt-4-11.json",
    "estimates-gpt-4-12.json",
    "estimates-gpt-4-13.json",
    "estimates-gpt-4-14.json",
    "estimates-gpt-4-15.json",
    "estimates-gpt-4-16.json",
    "estimates-gpt-4-17.json",
    "estimates-gpt-4-18.json",
]

median_errors = []
max_errors = []
for estimate_file in estimate_files:
    estimates = json.load(open(estimate_file))
    errors = []
    for result in estimates:
        error = abs(float(result["real"]) - float(result["estimate"])) / float(
            result["real"]
        )
        errors.append(error)
    median_errors.append(np.median(errors))
    max_errors.append(np.max(errors))

num_examples = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

plt.plot(num_examples, median_errors, "o-")
plt.xlabel("Number of examples")
plt.ylabel("Median Relative error")
plt.savefig("median_estimate_error_vs_num_examples.png")
plt.close()

plt.plot(num_examples, max_errors, "o-")
plt.xlabel("Number of examples")
plt.ylabel("Max Relative error")
plt.savefig("max_estimate_error_vs_num_examples.png")
plt.close()
