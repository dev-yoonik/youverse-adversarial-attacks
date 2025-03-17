import random


def part1():
    # Read the original labels
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\identity_CelebA.txt", "r") as f:
        lines = [line.strip().split() for line in f.readlines()]

    # Assign a random label (0 or 1) to each image while maintaining order
    attack_labels = [(img, str(random.randint(0, 1))) for img, _ in lines]

    # Write to the new file
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\attack_CelebA.txt", "w") as f:
        for img, label in attack_labels:
            f.write(f"{img} {label}\n")

    print("attack_label_path.txt generated successfully.")


def part2():

    # Read the original labels from label.txt.
    # Each line has the format: image_name identity
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\identity_CelebA.txt", "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    # Create a mapping from image name to identity.
    image_to_identity = {img: identity for img, identity in lines}

    # Build a dictionary mapping each identity to the list of images with that identity.
    id_to_images = {}
    for img, identity in lines:
        id_to_images.setdefault(identity, []).append(img)

    # Create a list of all image names.
    all_images = [img for img, _ in lines]

    # Precompute a candidate map for impersonation: for each identity,
    # store all images from different identities.
    candidate_map = {}
    for identity in id_to_images:
        # List all images whose identity is not equal to the current one.
        candidate_map[identity] = [img for img in all_images if image_to_identity[img] != identity]

    # Read the attack labels from attack_label_path.txt.
    # Each line has the format: image_name attack_label
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\attack_CelebA.txt", "r") as f:
        attack_labels = [line.strip().split() for line in f if line.strip()]

    # Write the final output to a file.
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\bonafide_evasion_impersonation.txt", "w") as f:
        # Write header (optional)
        f.write("original_image original_identity attack_type attack_image\n")

        # Process each image while preserving order.
        for img, att_label in attack_labels:
            orig_identity = image_to_identity.get(img, "unknown")

            if att_label == "0":
                # Bonafide image: no attack.
                attack_type = "bonafide"
                attack_img = img
            else:
                # For attack label 1, randomly choose evasion or impersonation.
                if random.random() < 0.5:
                    attack_type = "evasion"
                    attack_img = img
                else:
                    attack_type = "impersonation"
                    # Quickly retrieve candidates that have a different identity.
                    candidates = candidate_map.get(orig_identity, [])
                    if candidates:
                        attack_img = random.choice(candidates)
                    else:
                        # Fallback if no candidate exists.
                        attack_img = img

            # Write the result.
            f.write(f"{img} {orig_identity} {attack_type} {attack_img}\n")

    print("final_attack_output.txt generated successfully.")


def part3():
    import random

    # Number of attack splits/groups you want.
    N = 9  # <-- Change this to the number of attack splits you need.

    # Read the final attack file generated previously.
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\bonafide_evasion_impersonation.txt", "r") as f:
        lines = f.readlines()

    # Check if there's a header; if so, skip it.
    if lines[0].strip().startswith("original_image"):
        header = lines[0].strip().split()
        data_lines = lines[1:]
    else:
        data_lines = lines

    # Separate bonafide images from attacked ones.
    bonafide_list = []  # Will store lines for bonafide images.
    attacked_evasion = []  # For attacked images with evasion.
    attacked_impersonation = []  # For attacked images with impersonation.

    for line in data_lines:
        parts = line.strip().split()
        if not parts:
            continue
        # Expected columns: original_image, original_identity, attack_type, attack_image
        orig_img, orig_id, attack_type, attack_img = parts
        if attack_type == "bonafide":
            bonafide_list.append(line.strip())
        elif attack_type == "evasion":
            attacked_evasion.append(line.strip())
        elif attack_type == "impersonation":
            attacked_impersonation.append(line.strip())

    # To guarantee each attack gets half evasion and half impersonation,
    # we match the counts by taking the minimum of the two.
    count_evasion = len(attacked_evasion)
    count_impersonation = len(attacked_impersonation)
    common_count = min(count_evasion, count_impersonation)

    # Shuffle and trim to the common count.
    random.shuffle(attacked_evasion)
    random.shuffle(attacked_impersonation)
    attacked_evasion = attacked_evasion[:common_count]
    attacked_impersonation = attacked_impersonation[:common_count]

    # Now, split each list into N equal parts.
    # We'll use integer division (any remainder will be dropped).
    images_per_attack = common_count // N

    split_evasion = []
    split_impersonation = []
    for i in range(N):
        start = i * images_per_attack
        end = start + images_per_attack
        split_evasion.append(attacked_evasion[start:end])
        split_impersonation.append(attacked_impersonation[start:end])

    # Combine the corresponding splits so that each attack group has half evasion and half impersonation.
    attack_groups = []
    for i in range(N):
        group = split_evasion[i] + split_impersonation[i]
        random.shuffle(group)  # Optionally shuffle within the attack group.
        attack_groups.append(group)

    # Write the output: We'll output one file with an extra column for attack_id.
    # For bonafide images, we use "none" as the attack_id.
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\attack_splits.txt", "w") as f_out:
        # Write a header (optional)
        f_out.write("original_image original_identity attack_type attack_image attack_id\n")

        # Write bonafide images (attack_id marked as 'none')
        for line in bonafide_list:
            f_out.write(line + " none\n")

        # Write attacked images with their assigned attack id.
        for attack_id, group in enumerate(attack_groups):
            for line in group:
                f_out.write(line + f" {attack_id}\n")

    print("final_attack_with_attackid.txt generated successfully.")

def part4():
    import random

    # Number of attack splits/groups you want.
    N = 10  # Change this value to set the number of attack groups.

    # Read the final attack file (generated previously)
    with open(r"D:\datasets\adversarial\img_align_celeba\Anno\bonafide_evasion_impersonation.txt", "r") as f:
        lines = f.readlines()

    # Determine if the file has a header
    if lines[0].strip().startswith("original_image"):
        header = lines[0].strip()
        data_lines = lines[1:]
    else:
        header = None
        data_lines = lines

    # Separate bonafide images from attacked ones.
    # Expected columns in each line: original_image, original_identity, attack_type, attack_image
    bonafide_list = []
    attacked_evasion = []
    attacked_impersonation = []

    for line in data_lines:
        parts = line.strip().split()
        if not parts:
            continue
        # In case there's an extra column (attack_id), we only take the first 4 fields.
        orig_img, orig_id, attack_type, attack_img = parts[:4]

        if attack_type.lower() == "bonafide":
            bonafide_list.append(line.strip())
        elif attack_type.lower() == "evasion":
            attacked_evasion.append(line.strip())
        elif attack_type.lower() == "impersonation":
            attacked_impersonation.append(line.strip())

    # To ensure each attack group has an equal number of evasion and impersonation images,
    # we take the minimum count.
    common_count = min(len(attacked_evasion), len(attacked_impersonation))
    random.shuffle(attacked_evasion)
    random.shuffle(attacked_impersonation)
    attacked_evasion = attacked_evasion[:common_count]
    attacked_impersonation = attacked_impersonation[:common_count]

    # Each attack group will use an equal number of images from each type.
    images_per_attack = common_count // N

    attack_groups = []
    for i in range(N):
        start = i * images_per_attack
        end = start + images_per_attack
        group_evasion = attacked_evasion[start:end]
        group_impersonation = attacked_impersonation[start:end]
        # Combine the two halves
        group = group_evasion + group_impersonation
        random.shuffle(group)  # Optional: shuffle within the group.
        attack_groups.append(group)

    # Write each attack group to a separate file.
    for attack_id, group in enumerate(attack_groups):
        filename = f"final_attack_attackid_{attack_id}.txt"
        with open(filename, "w") as f_out:
            # Optionally include header with an extra column for attack id.
            if header:
                f_out.write(header + " attack_id\n")
            for line in group:
                f_out.write(line + f" {attack_id}\n")
        print(f"{filename} generated successfully.")

    # Write bonafide images to a separate file.
    with open("final_bonafide.txt", "w") as f_bona:
        if header:
            f_bona.write(header + " attack_id\n")
        for line in bonafide_list:
            f_bona.write(line + " none\n")
    print("final_bonafide.txt generated successfully.")

part1()
part2()
part3()
part4()