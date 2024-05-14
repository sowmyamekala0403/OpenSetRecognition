import subprocess
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

output_dir = "grid_search_results_wsvm_for_openset"
validation_output_dir= "./validation_results"
datasets = "/home/harinath/Documents/Dataset/"
dataset_dir = "/home/harinath/Downloads/filtered_dataset.csv"
output_file = "./output_file"
os.makedirs(output_file, exist_ok=True)

def convert_dataset(file_path):
    df = pd.read_csv(file_path, header=None)

    # Step 2: Separate labels and features
    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]

    label_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13,
                     'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26}
    labels = labels.map(label_mapping)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 4: Filter X_train to remove specified classes
    classes_to_remove = [['A','E','I','O','U'],['Q','W','E','R','T','Y','U','I','O','P'],['J','G','H','I','O','N','M','S','D','W','E','Z','X','O','P'],['A','C','V','R','T','I','M','N','B','P','Q','W','Z','J','D','S','X','U','E','F']]
    for removal_list in classes_to_remove:
    	
        mask = ~y_train.isin([label_mapping[label] for label in removal_list])
#       mask = ~y_train.isin([label_mapping[label] for label in classes_to_remove])
        X_train_filtered = X_train[mask]
        y_train_filtered = y_train[mask]
        save_train_and_test_file_to = datasets +'dataset_for_' + str(len(removal_list)) + '_unknown_classes.libsvm'
        os.makedirs(save_train_and_test_file_to,exist_ok=True)
        # Step 5: Normalization/Scaling
        scaler = MinMaxScaler()
        X_train_normalized = scaler.fit_transform(X_train_filtered)
        X_test_normalized = scaler.transform(X_test)
        print(y_train_filtered.unique())
        # Step 6: Save as LIBSVM Format
        libsvm_train_file = save_train_and_test_file_to + '/train_file.libsvm'
        libsvm_test_file = save_train_and_test_file_to + '/test_file.libsvm'
        save_libsvm_file(X_train_normalized, y_train_filtered, libsvm_train_file)
        save_libsvm_file(X_test_normalized, y_test, libsvm_test_file)

    
def save_libsvm_file(data, labels, filename):
    with open(filename, 'w') as file:
        for label, features in zip(labels, data):
            line = f"{label} {' '.join([f'{i + 1}:{value}' for i, value in enumerate(features)])}\n"
            file.write(line)

convert_dataset(dataset_dir)
# Define parameters
gamma_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
tresholds = [0.027,0.0635,0.114,0.1935]

# Create output directory if it doesn't exist


# DataFrame to store validation results

gamma_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
output_dir = "grid_search_results_wsvm"
os.makedirs(output_dir, exist_ok=True)

# Function to train and validate model
def train_and_validate_model(train_file, test_file, output_dir,treshold):
    dataset_name = os.path.basename(train_file).split('.')[0]
    validation_results = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1_Score", "True_Positive", "True_Negative", "False_Positive", "False_Negative"])
    os.makedirs(output_dir, exist_ok=True)
    for gamma in gamma_values:
        for C in C_values:
            model_file = f"{output_dir}/wsvm_model_gamma_{gamma}_C_{C}.model"
            print(f"Training WSVM with gamma={gamma}, C={C}, and threshold=0.5 for {dataset_name}...")
            command = ["./libsvm-openset/svm-train", "-s", "8", "-t", "2", "-g", str(gamma), "-c", str(C), train_file, model_file]
            subprocess.run(command)

            # Validate model
            print(f"tetsing WSVM model {model_file} with -P = {treshold} and -C = 0.001")
            validation_output = subprocess.run(["./libsvm-openset/svm-predict", "-P", str(treshold), "-C","0.001",test_file, model_file,output_file+"/outputfor"+str(gamma)+str(C)+".txt"], capture_output=True, text=True)
            # Extract validation metrics
            accuracy = float(validation_output.stdout.split('Recognition Accuracy = ')[1].split('%')[0])
            precision = float(validation_output.stdout.split('Precision=')[1].split(',')[0].strip())
            recall = float(validation_output.stdout.split('Recall=')[1].split(' Fmeasure=')[0].strip())
            f1_score = float(validation_output.stdout.split('Fmeasure=')[1].split(' ')[0].strip())
            true_positive = int(validation_output.stdout.split('True pos ')[1].split(' ')[0])
            true_negative = int(validation_output.stdout.split('True Neg ')[1].split(' ')[0].replace(',',''))
            false_positive = int(validation_output.stdout.split('False Pos ')[1].split(' ')[0].replace(',',''))
            false_negative = int(validation_output.stdout.split('False neg ')[1].split(' ')[0].replace(',',''))
            
            data={"Model": model_file, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1_score, "True_Positive": true_positive, "True_Negative": true_negative, "False_Positive": false_positive, "False_Negative": false_negative}
            print(data)
            
            # Append results to DataFrame
            validation_results = pd.concat([validation_results, pd.DataFrame([data])], ignore_index=True)


    # Save validation results to CSV
    validation_results.to_csv(f"{output_dir}/validation_results_for_"+dataset_name+".csv", index=False)
    print("Validation results saved for", dataset_name)	    

# Iterate through each directory in dataset_dir
for root, dirs, files in os.walk(datasets):
    index=0
    for dir in dirs:
        train_file_path = os.path.join(root, dir, "train_file.libsvm")
        test_file_path = os.path.join(root, dir, "test_file.libsvm")
        if os.path.exists(train_file_path) and os.path.exists(test_file_path):
            output_dir = f"grid_search_results_wsvm_for_openset_{dir}"
            print(f"Processing dataset in {os.path.join(root, dir)}...")
            train_and_validate_model(train_file_path, test_file_path, output_dir,tresholds[index])
            index+=1
