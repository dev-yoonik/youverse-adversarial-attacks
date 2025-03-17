import pandas as pd

### PART 1 ###
def part1():
    # Path to the CSV file
    file_path = r"C:\Users\joaot\Downloads\similarities.csv"

    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"Number of entries: {len(df)}")

    # Get rows where similarities > 0.4
    df = df[df['similarity'] > 0.3]

    # Get lfw_path and celeba_path columns as lists
    lfw_paths = df['lfw_path'].tolist()
    celeba_paths = df['celeba_path'].tolist()
    similarities = df['similarity'].tolist()

    # Get number of images in each list
    lfw_count = len(lfw_paths)
    celeba_count = len(celeba_paths)

    print(f"Number of images in LFW: {lfw_count}")
    print(f"Number of images in CelebA: {celeba_count}")

    for i in range(lfw_count):
        print(lfw_paths[i], celeba_paths[i], similarities[i])


### PART 2 ###

def part2():
    """
    Angela_Merkel 9672
    Camilla_Parker_Bowles 1496
    Conan_OBrien 2049
    Dwayne_Wade 2747
    Emmit_Smith 2978
    Gabriel_Batistuta 4011
    Grant_Hackett 8329
    Kristin_Davis 5512
    Marieta_Chrousala 6353
    Michael_Schumacher 6803
    Roger_Clemens 4328
    Yekaterina_Guseva 2818
    """

    lfw_list = ["Angela_Merkel", "Camilla_Parker_Bowles", "Conan_OBrien", "Dwayne_Wade", "Emmit_Smith", "Gabriel_Batistuta", "Grant_Hackett", "Kristin_Davis", "Marieta_Chrousala", "Michael_Schumacher", "Roger_Clemens", "Yekaterina_Guseva"]
    celeba_list = ["9672", "1496", "2049", "2747", "2978", "4011", "8329", "5512", "6353", "6803", "4328", "2818"]

    # Check for pairs.txt how many of the lfw_list identities exist.
    lfw_count = 0
    with open(r"D:\datasets\adversarial\LFW\pairs.txt", "r") as f:
        for line in f:
            for name in lfw_list:
                if name in line:
                    lfw_count += 1
                    break
    print(f"lfw_count: {lfw_count}")

    # Check for list eval partition how many of the celeba_list identities exist in the training set.
    # Step 1 - Get the images in the training set

    celeba_count = 0
    train_files = []
    with open(r"D:\datasets\adversarial\img_align_celeba\Eval\list_eval_partition.txt", "r") as f:
        for line in f:
            if line.split(" ")[1].strip() == "0":
                train_files.append(line.split(" ")[0])

    overlap_names = []

    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\identity_CelebA.txt", "r") as f:
        for line in f:
            file_name = line.split(" ")[0]
            if file_name in train_files:
                for name in celeba_list:
                    if name == line.split(" ")[1].strip():
                        overlap_names.append(file_name)
                        celeba_count += 1
                        break
    print(f"celeba_count: {celeba_count}")

    # Write the overlap names to a file
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\celeba_lfw_train_overlap.txt", "w") as f:
        for name in overlap_names:
            f.write(f"{name}\n")


### PART 3 ###
# move files to eval
eval_list = []
with open(r"D:\datasets\adversarial\img_align_celeba\Anno\celeba_lfw_train_overlap.txt", "r") as f:
    for line in f:
        file = line.strip()
        eval_list.append(file)

with open(r"D:\datasets\adversarial\img_align_celeba\Eval\list_eval_partition.txt", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.split(" ")[0].strip() in eval_list:
        new_line = line.split(" ")[0].strip() + " 1\n"
    else:
        new_line = line
    new_lines.append(new_line)

with open(r"D:\datasets\adversarial\img_align_celeba\Eval\list_eval_partition_no_overlap.txt", "w") as f:
    for line in new_lines:
        f.write(line)