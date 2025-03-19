![logo](https://yk-website-images.s3.eu-west-1.amazonaws.com/LogoV4_TRANSPARENT.png?)

# README #

This repository aggregates several adversarial attacks adapted for face recognition purposes.

### What is this repository for? ###

Launching adversarial attacks on face images. 
These attacks can be made with evasion intent or impersonation intent. 
The code has 10 available attacks at the moment. The list is provided bellow:

- Carlini and Wagner (CW) attack
- Deepfool attack
- DI2FGSM attack
- Evolutionary attack
- IFGSM attack
- JSMA attack
- LBFGS attack
- MIFGSM attack
- PI-FGSM attack
- TI-FGSM attack

### How do I get set up? ###

#### Attack a dataset: ####

Install dependencies:

    pip install -r requirements.txt

Configuration:

Set the configuration file in the config.yaml file. Example:
    
    defaults:
      - attack: di2fgsm
      - model: insightface_r50
      - dataset_attacker: lfw_attacker

    save_root: "D:/results/LFW_attacks"
    save_dir: "${.save_root}/${basename:${dataset_attacker.dataset_root}}_attack/${attack_name:${attack._target_}}/Run_${now:%Y-%m-%d-%H-%M-%S}"
    hydra:
      run:
        dir: ${save_dir}

Each attack, model and attacker have their own configuration file.
Each possible parameter that can be altered is defined in the config_schemas folder.

After setting up the attack, dataset attacker and model configs, simply run:
    
    python attack.py

A new folder will be generated with the attacked images and logs.

##### Model #####

We make available a single [open source FR model](https://1drv.ms/u/c/5fcfe07ba7cf0300/ETcJlEC23OJPjtOEiRIgZwMB-xeBMgV7hkqQpxrhrJp5KA?e=j8c7AL) based on the insightface architecture which can be used with the insightface model config.

#### Use in own pipeline: ####

The attacks were implemented in the albumentations format, a common image augmentation library. As such,
if you import the fr_attacks folder you can use each attack as a normal augmentation.
    
    import cv2
    from fr_attacks import I_FGSMAttack

    attack_transform = I_FGSMAttack(params)
    face_image_1 = cv2.imread("face_image_1.jpg")
    face_image_2 = cv2.imread("face_image_2.jpg")

    attacked = attack_transform(image=face_image_1, target=face_image_2, targeted=True)["image"]


### Contacts ##

adversarial@youverse.id