import re
import sys
import matplotlib.pyplot as plt

# Function to extract epochs and loss values from the log file
def extract_epochs_losses(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.readlines()
    
    # Patterns to match lines containing epoch and loss information
    pattern = re.compile(r'(\d+)\s*-\s*(\d+\.\d+e?-?\d*)')
    
    # Extract all matches
    losses = []
    epochs = []
    for line in log_content:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)
    
    return epochs, losses

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_loss.py <path_to_log_file>")
        sys.exit(1)
    
    log_file_path = sys.argv[1]

    # Extract the epochs and loss values
    epochs, loss_values = extract_epochs_losses(log_file_path)

    # Debugging output to verify extraction
    print(f"Extracted {len(epochs)} epochs and {len(loss_values)} loss values.")
    print("Sample epochs:", epochs[:5])
    print("Sample loss values:", loss_values[:5])

    # Plotting the loss chart
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Distillation Loss Curve: MobileSAM Teacher to Efficientvit-B0')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

